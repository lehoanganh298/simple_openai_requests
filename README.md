# Simple OpenAI Request

This script encompass these OpenAI request usecases:

- Calling multiple synchronous request in parallel (with retrying when rate limit hit)
- Making batch request (with checking status, pulling output file and merging result)
- Exact match prompt caching
  into a simple request interface
  `make_openai_request(conversations, use_batch, use_cache, cache_file='~/.gpt_cache.json',batch_dir='~/.gpt_batch_requests',num_threads=10)`

Explain:

- conversations: list of conversations/messages
  - Example: [
    [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
    ],
    [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How to make an OpenAI API request"}
    ]
    ]
- use_batch: 
	- use_batch=True: use batch API
	- use_batch=False: use synchronous API
- use_cache: whether to use caching. Cache can be used in both synchronous and batch api.
- cache_file: the cache file, if ignore, it would be `~/.gpt_cache.json`
- batch_dir: the directory to save files for making batch API requests, default is `~/.gpt_batch_requests`
- num_threads: number of parallel threads when using synchronous API
