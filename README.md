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

### Additional Options:

- `generation_args`: Additional arguments for the API call (e.g., max_tokens, temperature)
- `cache_file`: Path to the cache file (default: '~/.gpt_cache.json')
- `batch_dir`: Directory for batch processing files (default: '~/.gpt_batch_requests')
- `full_response`: Return full API response or just the message content

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