import concurrent.futures
from openai import OpenAI
from typing import List, Dict, Any, Union
from tqdm import tqdm
import logging
import time
import json
import os
import openai

from caching import load_cache, save_cache, get_cache_key

DEFAULT_MAX_RETRIES = 10
DEFAULT_RETRY_DELAY = 30
DEFAULT_GENERATION_ARGS = {}
DEFAULT_MAX_WORKERS = 10
CACHE_SAVE_INTERVAL = 5  # Save cache to file every x updates

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

def truncate_conversation(conversation: List[Dict[str, str]], max_length: int = 100) -> str:
    """Truncate the conversation to a specified maximum length."""
    conv_str = json.dumps(conversation)
    return conv_str[:max_length] + ('...' if len(conv_str) > max_length else '')


def make_api_call(client: OpenAI, 
                  conversation: List[Dict[str, str]], 
                  model: str, 
                  generation_args: Dict[str, Any] = DEFAULT_GENERATION_ARGS, 
                  max_retries: int = DEFAULT_MAX_RETRIES, 
                  retry_delay: float = DEFAULT_RETRY_DELAY, 
                  index: int = None) -> Dict[str, Any]:
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=conversation,
                **generation_args
            )
            
            return {
                "index": index,
                "conversation": conversation,
                "response": json.loads(completion.json()),
                "error": None
            }
        except Exception as e:
            error_message = str(e)
            if "rate limit" in error_message.lower():
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limit hit. Retrying in {retry_delay} seconds. Attempt {attempt + 1}/{max_retries}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Rate limit exceeded: {truncate_conversation(conversation)}")
                    return {
                        "index": index,
                        "conversation": conversation,
                        "response": None,
                        "error": "Rate limit exceeded"
                    }
            else:
                logger.error(f"{error_message}: {truncate_conversation(conversation)}")
                return {
                    "index": index,
                    "conversation": conversation,
                    "response": None,
                    "error": error_message
                }
    
    # This should never be reached, but just in case
    logger.error(f"Unexpected error occurred: {truncate_conversation(conversation)}")
    return {
        "index": index,
        "conversation": conversation,
        "response": None,
        "error": "Unexpected error occurred"
    }


def make_parallel_sync_requests(client: OpenAI, 
                                conversations: Union[List[List[Dict[str, str]]], List[Dict[str, Any]]], 
                                model: str, 
                                generation_args: Dict[str, Any] = DEFAULT_GENERATION_ARGS, 
                                max_workers: int = DEFAULT_MAX_WORKERS, 
                                max_retries: int = DEFAULT_MAX_RETRIES, 
                                retry_delay: float = DEFAULT_RETRY_DELAY, 
                                use_cache: bool = False,
                                cache=None, 
                                cache_file=None) -> List[Dict[str, Any]]:
    num_requests = len(conversations)
    
    # logger.info(f"Preparing to make {num_requests} API requests")
    # logger.info(f"Model: {model}")
    
    cache_update_count = 0

    # Normalize conversations to include index
    if isinstance(conversations[0], list):
        conversations = [{"index": idx, "conversation": conv} for idx, conv in enumerate(conversations)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(make_api_call, client, item["conversation"], model, generation_args, max_retries, retry_delay, item["index"])
            for item in conversations
        ]
        
        results = []
        with tqdm(total=num_requests, desc="API Requests", unit="request") as pbar:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                
                # Update cache if the request was successful
                if result["error"] is None and use_cache:
                    cache_key = get_cache_key(result["conversation"], model, generation_args)
                    cache[cache_key] = {"model": model, "generation_args": generation_args, 'conversation': result['conversation'], 'response': result['response']}
                    cache_update_count += 1

                    # Save cache to file every N updates
                    if cache_update_count % CACHE_SAVE_INTERVAL == 0 and cache_file:
                        save_cache(cache, cache_file)

                results.append(result)
                pbar.update(1)

    # Save cache to file after finishing all requests
    if use_cache:
        save_cache(cache, cache_file)

    # Sort results to match the order of input conversations
    sorted_results = sorted(results, key=lambda x: x["index"])
    
    logger.info(f"Completed {num_requests} synchronous API requests")
    successful_requests = sum(1 for r in sorted_results if r["error"] is None)
    failed_requests = sum(1 for r in sorted_results if r["error"] is not None)
    logger.info(f"Successful requests: {successful_requests}/{num_requests}")
    logger.info(f"Failed requests: {failed_requests}/{num_requests}")

    return sorted_results

# Example usage
if __name__ == "__main__":
    conversations = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        [
            {"role": "system", "content": "You are a friendly chatbot."},
            {"role": "user", "content": "How are you?"}
        ],
        [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What's 2 + 2?"}
        ]
    ]
    
    model = "gpt-4o-mini"
    generation_args = {
        "max_tokens": 150,
        "temperature": 0.7
    }
    max_workers = 3
    max_retries = 3  # Maximum number of retries for rate limit errors
    retry_delay = 5  # 5 seconds delay between retries
    
    results = make_parallel_sync_requests(conversations, model, generation_args, max_workers, max_retries, retry_delay)
    
    for result in results:
        print(f"\nResult {result['index'] + 1}:")
        print("Messages:")
        for message in result["conversation"]:
            print(f"  {message['role']}: {message['content']}")
        if result["response"]:
            print("Response:")
            print(f"  {result['response'].role}: {result['response'].content}")
        if result["error"]:
            print(f"Error: {result['error']}")