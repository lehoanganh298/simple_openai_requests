import json
import time
import os
import logging
from openai import OpenAI
from typing import List, Dict, Any, Tuple, Union
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

STATUS_CHECK_INTERVAL = 60
MAX_REQUESTS_PER_BATCH = 50000
MAX_BATCH_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_request(conversation: List[Dict[str, str]], model_name: str, idx: int, generation_args: Dict[str, Any] = {}) -> Dict[str, Any]:
    return {
        "custom_id": f"request-{idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model_name,
            "messages": conversation,
            **generation_args
        }
    }

def estimate_request_size(conversation: List[Dict[str, str]], model_name: str, idx: int) -> int:
    request = create_request(conversation, model_name, idx)
    return len(json.dumps(request).encode('utf-8')) + 1

def create_batch_request_file(batch: List[Dict[str, Any]], model_name: str, file_path: str, generation_args: Dict[str, Any] = {}) -> None:
    # logger.info(f"Creating batch request file at: {file_path}")
    with open(file_path, 'w') as f:
        for item in batch:
            request = create_request(item["conversation"], model_name, item["index"], generation_args)
            f.write(json.dumps(request) + '\n')
    logger.info(f"Batch request file created: {file_path}")

def process_batch_output(output_file: str, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = {item["index"]: {"index": item["index"], "conversation": item["conversation"], "response": None, 'error': None} for item in batch}

    with open(output_file, 'r') as f:
        for line in f:
            result = json.loads(line)
            custom_id = result['custom_id']
            idx = int(custom_id.split('-')[1])
            if idx in results:
                if result.get('error') is not None:
                    results[idx]['error'] = result.get('error')
                else:
                    results[idx]["response"] = result['response']['body']

    logger.info(f"Batch output processed: {output_file}")

    return list(results.values())

def process_single_batch(client: OpenAI, batch: List[Dict[str, Any]], model_name: str, batch_dir: str, batch_run_name: str, generation_args: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
    batch_request_file = f"{batch_dir}/{batch_run_name}_batch_request.jsonl"
    create_batch_request_file(batch, model_name, batch_request_file, generation_args)

    with open(batch_request_file, 'rb') as f:
        file = client.files.create(file=f, purpose="batch")

    logger.info(f"Batch submitted with file ID: {file.id}")
    
    batch_job = client.batches.create(
        input_file_id=file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    logger.info(f"Batch processing started. Batch ID: {batch_job.id}")
    time.sleep(3)

    batch_in_progress = False
    while True:
        batch_status = json.loads(client.batches.retrieve(batch_job.id).json())
        total_requests = batch_status['request_counts']['total']
        completed_requests = batch_status['request_counts']['completed']

        if batch_status['status'] == "in_progress":
            if not batch_in_progress:
                batch_in_progress = True
                pbar = tqdm(total=total_requests, ncols=100, desc="Processing batch")
            pbar.update(completed_requests - pbar.n)

        elif batch_status['status'] == "completed":
            if batch_in_progress:
                pbar.update(total_requests - pbar.n)
            break
        elif batch_status['status'] in ["failed", "expired"]:
            logger.error(f"Batch failed or expired. Status: {batch_status['status']}")
            if batch_in_progress:
                pbar.close()
            return []
        else:
            logger.info(f"Batch status: {batch_status['status']}")

        
        time.sleep(STATUS_CHECK_INTERVAL)
    if batch_in_progress:
        pbar.close()

    output_file_content = client.files.content(batch_status['output_file_id'])
    batch_output_file = f"{batch_dir}/{batch_run_name}_batch_output.jsonl"

    with open(batch_output_file, 'w') as f:
        f.write(output_file_content.text)

    logger.info(f"Batch output file saved: {batch_output_file}")

    return process_batch_output(batch_output_file, batch)

def make_batch_request(client: OpenAI, conversations: Union[List[List[Dict[str, str]]], List[Dict[str, Any]]], model_name: str, batch_dir: str, batch_run_name: str, generation_args: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
    """
    Make a OpenAI batch requests for a list of conversations, retrieve and return the responses

    This function checks if the batch is within the allowed limits for number of requests and total size of a single batch,
    then processes the batch if it meets the criteria.

    Args:
        client (OpenAI): The OpenAI client instance.
        conversations (Union[List[List[Dict[str, str]]], List[Dict[str, Any]]]): A list of conversations, where each conversation
            is either a list of message dictionaries or a dictionary with 'index' and 'conversation' keys.
        model_name (str): The name of the OpenAI model to use.
        batch_dir (str): The directory to store batch-related files.
        batch_run_name (str): A unique name for this batch run.
        generation_args (Dict[str, Any], optional): Additional arguments for text generation. Defaults to {}.

    Returns:
        List[Dict[str, Any]]: A list of processed results. Each result is a dictionary containing the original prompt,
                              the API response, and an index.

    Raises:
        None, but logs errors if limits are exceeded.

    Note:
        - The function creates the batch directory if it doesn't exist.
        - It checks against MAX_REQUESTS_PER_BATCH and MAX_BATCH_FILE_SIZE_BYTES limits.
        - If limits are exceeded, it logs an error and returns an empty list.
    """
    os.makedirs(batch_dir, exist_ok=True)
    
    # Normalize conversations to include index
    if isinstance(conversations[0], list):
        conversations = [{"index": idx, "conversation": conv} for idx, conv in enumerate(conversations)]
    
    # Check if the total number of conversations exceeds the limit
    if len(conversations) > MAX_REQUESTS_PER_BATCH:
        logger.error(f"Number of conversations ({len(conversations)}) exceeds the maximum limit of {MAX_REQUESTS_PER_BATCH}")
        return []

    # Estimate the total size of all conversations
    total_estimated_size = sum(estimate_request_size(conv["conversation"], model_name, conv["index"]) for conv in conversations)
    
    if total_estimated_size > MAX_BATCH_FILE_SIZE_BYTES:
        logger.error(f"Estimated total size ({total_estimated_size} bytes) exceeds the maximum limit of {MAX_BATCH_FILE_SIZE_BYTES} bytes")
        return []

    # If we're here, it means we're within limits, so we can process the single batch
    batch = conversations
    results = process_single_batch(client, batch, model_name, batch_dir, batch_run_name, generation_args)

    return results


##############################################################################
# Multiple batch requests

def submit_and_process_batch(client: OpenAI, batch_file: str, batch_num: int) -> List[Dict[str, Any]]:
    with open(batch_file, 'rb') as f:
        file = client.files.create(file=f, purpose="batch")

    logger.info(f"Batch {batch_num} submitted with file ID: {file.id}")
    
    batch_job = client.batches.create(
        input_file_id=file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    logger.info(f"Batch {batch_num} processing started. Batch ID: {batch_job.id}")
    
    while True:
        batch_status = json.loads(client.batches.retrieve(batch_job.id).json())
        total_requests = batch_status['request_counts']['total']
        completed_requests = batch_status['request_counts']['completed']

        logger.info(f"Batch {batch_num} status: {batch_status['status']}")
        logger.info(f"Completed requests: {completed_requests}/{total_requests}")

        if batch_status['status'] == "completed":
            break
        elif batch_status['status'] in ["failed", "expired"]:
            logger.error(f"Batch {batch_num} failed or expired. Status: {batch_status['status']}")
            return []

        time.sleep(STATUS_CHECK_INTERVAL)

    output_file_content = client.files.content(batch_status['output_file_id'])
    batch_output_file = f"{os.path.dirname(batch_file)}/{os.path.basename(batch_file).replace('_request', '_output')}"

    with open(batch_output_file, 'w') as f:
        f.write(output_file_content.text)

    logger.info(f"Batch output file saved: {batch_output_file}")

    with open(batch_file, 'r') as f:
        batch = [{"index": int(json.loads(line)['custom_id'].split('-')[1]), "conversation": json.loads(line)['body']['messages']} for line in f]

    return process_batch_output(batch_output_file, batch)

def make_batch_request_multiple_batches(client: OpenAI, conversations: Union[List[List[Dict[str, str]]], List[Dict[str, Any]]], model_name: str, batch_dir: str, batch_run_name: str, generation_args: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
    """
    Make one or more OpenAI batch requests for a list of conversations, retrieve and return the responses
    (split into multiple batch if list of conversations too large for one batch file)

    Args:
        client (OpenAI): An instance of the OpenAI client.
        conversations (Union[List[List[Dict[str, str]]], List[Dict[str, Any]]]): A list of conversations, where each conversation
            is either a list of message dictionaries or a dictionary with 'index' and 'conversation' keys.
        model_name (str): The name of the OpenAI model to use for processing.
        batch_dir (str): The directory where batch files will be stored.
        batch_run_name (str): A unique name for this batch run, used in file naming.
        generation_args (Dict[str, Any], optional): Additional arguments for the API call. Defaults to {}.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the processed results for each conversation.
        Each dictionary has 'index', 'conversation', 'response' and 'error' keys.

    Raises:
        Exception: If an error occurs during batch processing.

    Note:
        - This function uses threading for concurrent processing of multiple batches.
        - It respects the MAX_REQUESTS_PER_BATCH and MAX_BATCH_FILE_SIZE_BYTES constants.
        - Progress is displayed using tqdm for both single and multiple batch processing.
    """
    os.makedirs(batch_dir, exist_ok=True)
    
    # Normalize conversations to include index
    if isinstance(conversations[0], list):
        conversations = [{"index": idx, "conversation": conv} for idx, conv in enumerate(conversations)]
    
    batches = []
    current_batch = []
    current_batch_size = 0

    for conv in conversations:
        estimated_size = estimate_request_size(conv["conversation"], model_name, conv["index"])
        
        if len(current_batch) >= MAX_REQUESTS_PER_BATCH or current_batch_size + estimated_size > MAX_BATCH_FILE_SIZE_BYTES:
            batches.append(current_batch)
            current_batch = []
            current_batch_size = 0

        current_batch.append(conv)
        current_batch_size += estimated_size

    if current_batch:
        batches.append(current_batch)

    if len(batches) == 1:
        # Process single batch with progress bar
        return process_single_batch(client, batches[0], model_name, batch_dir, batch_run_name, generation_args)
    else:
        # Process multiple batches concurrently
        batch_files = []
        for i, batch in enumerate(batches):
            batch_file = f"{batch_dir}/{batch_run_name}_batch_{i}_request.jsonl"
            create_batch_request_file(batch, model_name, batch_file, generation_args)
            batch_files.append(batch_file)

        all_results = []
        with ThreadPoolExecutor(max_workers=min(len(batch_files), 10)) as executor:
            future_to_batch = {executor.submit(submit_and_process_batch, client, batch_file, i): i 
                               for i, batch_file in enumerate(batch_files)}
            
            for future in tqdm(as_completed(future_to_batch), total=len(batch_files), desc="Processing batches"):
                batch_num = future_to_batch[future]
                # try:
                batch_results = future.result()
                all_results.extend(batch_results)
                logger.info(f"Batch {batch_num} completed successfully")
                # except Exception as e:
                #     logger.error(f"Batch {batch_num} generated an exception: {str(e)}")

        all_results.sort(key=lambda x: x['index'])
        logger.info(f"Batch request completed with {len(uncached_results)} results")
        
        return all_results

# Example usage
if __name__ == "__main__":
    client = OpenAI()
    conversations = [
        {"index": i, "conversation": [{"role": "user", "content": f"This is a conversation with varying content length {i*100}"}]} for i in range(60000)
    ]
    model_name = "gpt-3.5-turbo-0125"
    batch_dir = "batch_outputs"
    batch_run_name = "efficient_batching_test_run"

    try:
        results = make_batch_request(client, conversations, model_name, batch_dir, batch_run_name)
        print(f"Processed {len(results)} conversations")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")