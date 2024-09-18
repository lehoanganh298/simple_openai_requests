import json
import os

def load_cache(cache_file):
    cache_file = os.path.expanduser(cache_file)
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return {get_cache_key(item['conversation'], item['model'], item['generation_args']): item for item in json.load(f)}
    return {}

def save_cache(cache, cache_file):
    cache_file = os.path.expanduser(cache_file)
    cache_list = [{'conversation': value['conversation'], 
                   'model': value['model'], 
                   'generation_args': value['generation_args'], 
                   'response': value['response']} for key, value in cache.items()]
    with open(cache_file, 'w') as f:
        json.dump(cache_list, f, ensure_ascii=False, indent=2)

def get_cache_key(conversation, model_name, generation_args={}):
    return json.dumps({'conversation': conversation, 'model': model_name, 'generation_args': generation_args})
