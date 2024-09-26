import os
import json
import uuid
import logging
from openai import OpenAI
from simple_openai_requests.batch_requests import make_batch_request_multiple_batches
from simple_openai_requests.synchronous_requests import make_parallel_sync_requests
from simple_openai_requests.caching import load_cache, save_cache, get_cache_key
from typing import List, Dict, Any, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def make_openai_requests(
    conversations: Union[List[str], List[List[Dict[str, str]]], List[Dict[str, Any]]],
    model: str,
    use_batch: bool = False,
    use_cache: bool = True,
    generation_args: Dict[str, Any] = {},
    cache_file: str = None,
    batch_dir: str = None,
    batch_run_name: str = None,
    max_workers: int = 10, 
    max_retries: int = 10, 
    retry_delay: int = 30, 
    status_check_interval: int = 60,
    full_response: bool = False,
    api_key = None,
    user_confirm: bool = True,
    base_url: str = None,  # New parameter for base URL
) -> List[Dict[str, Any]]:
    """
    Make OpenAI API requests for a list of conversations, with options for batching API 
    or parallel synchronous API, with or without caching.

    This function processes a list of conversations, sending them to the OpenAI API
    either individually or in batches. It supports caching to avoid redundant API calls
    and can use either synchronous or batch processing methods.

    Args:
        conversations (Union[List[str], List[List[Dict[str, str]]], List[Dict[str, Any]]]): 
            A list of conversations in one of the following formats:
            - List[str]: Each string is a single message/prompt.
            - List[List[Dict[str, str]]]: Each inner list is a conversation, where each
              dictionary has 'role' and 'content' keys.
            - List[Dict[str, Any]]: Each dictionary contains 'index' and 'conversation' keys,
              where 'conversation' is a list of message dictionaries.
        model (str): The name of the OpenAI model to use (e.g., "gpt-3.5-turbo").
        use_batch (bool, optional): If True, use batch processing. Default is False.
        use_cache (bool, optional): If True, use caching to avoid redundant API calls. Default is True.
        generation_args (Dict[str, Any]): Additional arguments for the API call, such as
            max_tokens, temperature, etc. Default is empty.
        cache_file (str, optional): Path to the cache file. If not set, it will check the 
            environment variable SIMPLE_OPENAI_REQUESTS_CACHE_FILE. If that is also not set, 
            it defaults to '~/.gpt_cache.pkl'.
        batch_dir (str, optional): Directory for batch processing files. If not set, it will 
            check the environment variable SIMPLE_OPENAI_REQUESTS_BATCH_DIR. If that is also 
            not set, it defaults to '~/.gpt_batch_requests'.
        batch_run_name (str, optional): A unique identifier for the batch run. If not provided
            and use_batch is True, a UUID will be generated.
        max_workers (int, optional): Maximum number of worker threads for parallel processing. Default is 10.
        max_retries (int, optional): Maximum number of retries for failed API calls. Default is 10.
        retry_delay (int, optional): Delay in seconds between retries. Default is 30.
        status_check_interval (int, optional): Interval in seconds between status checks for batch requests. Default is 60.
        full_response (bool, optional): If True, return the full response object. 
            If False, return only the message content if available. Default is False.
        api_key (str, optional): OpenAI API key. If not provided, it will be read from the OPENAI_API_KEY environment variable.
        user_confirm (bool, optional): If True, prompt for user confirmation before making API requests. 
            If False, bypass user confirmation. Default is True.
        base_url (str, optional): Base URL for the OpenAI client. Use when calling vLLM APIs.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing:
            - 'index': index of the conversation
            - 'conversation': The original conversation
            - 'response': The API response
            - 'is_cached_response': Boolean indicating if the response was from cache
            - 'error': Any error message (None if no error occurred)

    Raises:
        ValueError: If OPENAI_API_KEY is not set in environment variables and api_key is not provided.

    Note:
        - The function requires the OPENAI_API_KEY to be set in the environment variables or provided as api_key parameter.
        - When using caching, responses are saved to and loaded from the specified cache file.
        - Batch processing is more efficient for large numbers of conversations but may have
          a higher latency for individual responses.
        - The max_workers, max_retries, and retry_delay parameters are used for parallel synchronous requests.
    """
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    if api_key is None:
        logger.error("API key not found")
        raise ValueError("Either api_key param or OPENAI_API_KEY environment variable need to be set")
    
    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        client = OpenAI(api_key=api_key)

    # Set cache_file and batch_dir to environment variables if not provided
    cache_file = cache_file or os.getenv('SIMPLE_OPENAI_REQUESTS_CACHE_FILE', os.path.expanduser('~/.gpt_cache.pkl'))
    batch_dir = batch_dir or os.getenv('SIMPLE_OPENAI_REQUESTS_BATCH_DIR', os.path.expanduser('~/.gpt_batch_requests'))

    conversation_formated = reformat_conversations(conversations)
    if use_cache:
        cache, cached_results, uncached_conversations = check_cache(cache_file, conversation_formated, model, generation_args)
    else:
        logger.info("Cache disabled")
        cached_results = []
        cache = {}
        uncached_conversations = conversation_formated

    # Process uncached conversations
    if uncached_conversations:
        if use_batch and batch_run_name is None:
            batch_run_name = str(uuid.uuid4())
        
        confirm_message = f"Making {len(uncached_conversations)} {'batch' if use_batch else 'synchronous'} API requests using model '{model}'" + (f" with batch run name '{batch_run_name}'" if use_batch else '')
        # User confirmation
        if user_confirm:
            confirmation = input(confirm_message + "? ([y]/n): ")
            if confirmation.lower() in ['n', 'no']:
                logger.info("User cancelled the operation")
                return cached_results  # Return only cached results if user cancels
        else:
            logger.info(confirm_message)

        if use_batch:
            batch_dir = os.path.expanduser(batch_dir)
            uncached_results = make_batch_request_multiple_batches(
                client, 
                uncached_conversations, 
                model, 
                batch_dir, 
                batch_run_name, 
                status_check_interval
            )
            
            if use_cache:
                update_cache(uncached_results, cache, model, generation_args, cache_file)
        else:
            uncached_results = make_parallel_sync_requests(client, 
                                                           uncached_conversations, 
                                                           model, 
                                                           generation_args, 
                                                           max_workers, max_retries, retry_delay, 
                                                           use_cache, cache, cache_file)

        for result in uncached_results:
            result['is_cached_response'] = False
    else:
        uncached_results = []
        logger.info("No uncached conversations to process")
    
    # Combine and sort results
    all_results = sorted(cached_results + uncached_results, key=lambda x: x['index'])
    if use_cache:
        logger.info(f"Request completed. Total results: {len(all_results)}")
    
    # Process results if full_response is False
    if not full_response:
        for result in all_results:
            if result['response'] is not None:
                try:
                    result['response'] = result['response']['choices'][0]['message']['content']
                except (KeyError, IndexError):
                    pass  # Keep the full response if the expected structure is not found
    
    return all_results

def reformat_conversations(conversations):
    reformatted = []
    for idx, conv in enumerate(conversations):
        if isinstance(conv, str):
            conv = [{"role": "user", "content": conv}]
        
        if not isinstance(conv, dict) or "index" not in conv:
            conv = {"index": idx, "conversation": conv}
        
        reformatted.append(conv)
    
    return reformatted

def check_cache(cache_file, conversations, model, generation_args):
    logger.info(f"Loading cache from {cache_file}")
    cache = load_cache(cache_file)

    # Check cache and prepare uncached conversations
    cached_results = []
    uncached_conversations = []
        
    for conversation in conversations:
        cache_key = get_cache_key(conversation['conversation'], model, generation_args)
        if cache_key in cache:
            cached_results.append({**conversation, "response": cache[cache_key]['response'], "error": None, "is_cached_response": True})
            continue
        uncached_conversations.append(conversation)
        
    logger.info(f"Cache hits: {len(cached_results)}, Uncached requests: {len(uncached_conversations)}")
        
    return cache, cached_results, uncached_conversations


def update_cache(uncached_results, cache, model, generation_args, cache_file):
    # logger.info("Updating cache with new results")
    for result in uncached_results:
        if result['error'] is None:
            cache_key = get_cache_key(result['conversation'], model, generation_args)
            cache[cache_key] = {
                "model": model,
                "generation_args": generation_args,
                'conversation': result['conversation'],
                'response': result['response']
            }
    save_cache(cache, cache_file)
    logger.info(f"Cache updated and saved to {cache_file}")

# Example usage
if __name__ == "__main__":
    conversations = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How to make an OpenAI API request"}
        ]
    ]
    
    results = make_openai_requests(
        conversations=conversations,
        model="gpt-4o-mini",
        generation_args={},
        # use_batch=True,
        use_cache=False,
        max_workers=2,
        status_check_interval=30,  # Example usage of the new parameter
    )
    
    for result in results:
        print(result)