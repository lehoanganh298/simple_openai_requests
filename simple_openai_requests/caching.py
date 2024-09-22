import pickle
import os
import json

def load_cache(cache_file):
    cache_file = os.path.expanduser(cache_file)
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache_list = pickle.load(f)
            return {get_cache_key(item['conversation'], item['model'], item['generation_args']): item for item in cache_list}
    return {}

def save_cache(cache, cache_file):
    cache_file = os.path.expanduser(cache_file)
    cache_list = [{'conversation': value['conversation'], 
                   'model': value['model'], 
                   'generation_args': value['generation_args'], 
                   'response': value['response']} for key, value in cache.items()]
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_list, f)

def get_cache_key(conversation, model_name, generation_args={}):
    return json.dumps({'conversation': conversation, 'model': model_name, 'generation_args': generation_args})
