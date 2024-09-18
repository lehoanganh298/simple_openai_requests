# Simple OpenAI Request

This script provides a simple interface for making OpenAI API requests with various use cases:

- **Synchronous requests**: parallel requests with retry when rate limit hit
- **Batch requests**: create, check and merge result of batch API requests
- **Caching**: exact match caching to avoid redundant API calls

## Main Function

`make_openai_requests(conversations, model, use_batch=False, use_cache=True, ...)`

### Key Parameters:

- `conversations`: List of conversations/messages (supports multiple formats)
- `model`: OpenAI model to use (e.g., "gpt-3.5-turbo")
- `use_batch`: Set to True for batch API, False for synchronous API
- `use_cache`: Enable/disable caching
- `api_key`: OpenAI API key (if not provided, it will be read from the environment variable `OPENAI_API_KEY`)

### Additional Options:

- `generation_args`: Additional arguments for the API call (e.g., max_tokens, temperature)
- `cache_file`: Path to the cache file (default: environment variable `SIMPLE_OPENAI_REQUESTS_CACHE_FILE` or default as `~/.gpt_cache.json`)
- `batch_dir`: Directory for batch processing files (default: environment variable `SIMPLE_OPENAI_REQUESTS_BATCH_DIR` or default as `~/.gpt_batch_requests`)
- `full_response`: Return full API response or just the message content
- `user_confirm`: If True, prompts for user confirmation before making API requests

and other parameters in function `make_openai_requests()`'s documentation.

### Return Format:

The function returns a list of dictionaries, where each dictionary contains:

- `index`: The index of the conversation
- `conversation`: The original conversation
- `response`: The API response (full response object if `full_response=True`, otherwise just the message content)
- `is_cached_response`: Boolean indicating if the response was from cache
- `error`: Any error message (None if no error occurred)

## Installation

### Option 1: Install using pip

You can install the Simple OpenAI Request package using pip:

```bash
pip install simple-openai-requests
```

### Option 2: Install from source

To install the package from source, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/lehoanganh298/simple_openai_requests.git
   ```

2. Navigate to the project directory:
   ```bash
   cd simple_openai_requests
   ```

3. Install the package:
   ```bash
   pip install .
   ```


## Usage Examples:

### 1. Simple string prompts

```python
from simple_openai_requests import make_openai_requests

conversations = [
    "What is the capital of France?",
    "How does photosynthesis work?"
]

results = make_openai_requests(
    conversations=conversations,
    model="gpt-3.5-turbo",
    use_batch=False,
    use_cache=True
)

for result in results:
    print(f"Question: {result['conversation'][0]['content']}")
    print(f"Answer: {result['response']}\n")
```
### 2. Conversation format
```python
conversations = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the best way to learn programming?"}
    ],
    [
        {"role": "system", "content": "You are a knowledgeable historian."},
        {"role": "user", "content": "Explain the significance of the Industrial Revolution."}
    ]
]
results = make_openai_requests(
    conversations=conversations,
    model="gpt-4",
    use_batch=True,
    use_cache=False,
    generation_args={"max_tokens": 150}
)
for result in results:
    print(f"Question: {result['conversation'][-1]['content']}")
    print(f"Answer: {result['response']}\n")
```
### 3. Indexed conversation format

```python
conversations = [
    {
        "index": 0,
        "conversation": [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "Explain the Pythagorean theorem."}
        ]
    },
    {
        "index": 1,
        "conversation": [
            {"role": "system", "content": "You are a creative writing assistant."},
            {"role": "user", "content": "Give me a writing prompt for a short story."}
        ]
    }
]
results = make_openai_requests(
    conversations=conversations,
    model="gpt-3.5-turbo",
    use_batch=False,
    use_cache=True,
    max_workers=2
)
for result in results:
    print(f"Index: {result['index']}")
    print(f"Question: {result['conversation'][-1]['content']}")
    print(f"Answer: {result['response']}\n")
```