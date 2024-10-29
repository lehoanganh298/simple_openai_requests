import sqlite3
import json
from typing import Dict, Any, Optional, List
import logging
import os
import hashlib
from contextlib import contextmanager

logger = logging.getLogger(__name__)

def get_cache_key(conversation, model_name, generation_args={}):
    key_data = json.dumps({'conversation': conversation, 'model': model_name, 'generation_args': generation_args})
    return hashlib.sha256(key_data.encode()).hexdigest()

class SQLiteCache:
    def __init__(self, db_path: str):
        """Initialize SQLite cache with the given database path."""
        self.db_path = os.path.expanduser(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema if it doesn't exist."""
        with self._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    generation_args TEXT NOT NULL,
                    conversation TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Optional: Create an index on model for faster lookups
            # conn.execute('CREATE INDEX IF NOT EXISTS idx_model ON cache(model)')

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single cached response."""
        return self.get_many([cache_key]).get(cache_key)

    def get_many(self, cache_keys: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Retrieve multiple cached responses in a single database operation.
        
        Args:
            cache_keys: List of cache keys to retrieve
            
        Returns:
            Dictionary mapping cache keys to their cached values (or None if not found)
        """
        try:
            with self._get_connection() as conn:
                # Use parameterized query with multiple placeholders
                placeholders = ','.join('?' * len(cache_keys))
                query = f'SELECT cache_key, model, generation_args, conversation, response FROM cache WHERE cache_key IN ({placeholders})'
                
                cursor = conn.execute(query, cache_keys)
                results = cursor.fetchall()
                
                # Create result dictionary with all keys initially set to None
                cache_dict = {key: None for key in cache_keys}
                
                # Update with found results
                for row in results:
                    cache_dict[row[0]] = {
                        'model': row[1],
                        'generation_args': json.loads(row[2]),
                        'conversation': json.loads(row[3]),
                        'response': json.loads(row[4])
                    }
                
                return cache_dict
        except Exception as e:
            logger.error(f"Error retrieving multiple items from cache: {e}")
            return {key: None for key in cache_keys}

    def set(self, cache_key: str, value: Dict[str, Any]) -> bool:
        """Store a single response in the cache."""
        return self.set_many({cache_key: value})

    def set_many(self, items: Dict[str, Dict[str, Any]]) -> bool:
        """Store multiple responses in the cache in a single database operation.
        
        Args:
            items: Dictionary mapping cache keys to their values
            
        Returns:
            bool: True if successful, False if error occurred
        """
        try:
            with self._get_connection() as conn:
                # Prepare batch of values for insertion
                values = [
                    (
                        cache_key,
                        value['model'],
                        json.dumps(value['generation_args']),
                        json.dumps(value['conversation']),
                        json.dumps(value['response'])
                    )
                    for cache_key, value in items.items()
                ]
                
                conn.executemany(
                    'INSERT OR REPLACE INTO cache (cache_key, model, generation_args, conversation, response) VALUES (?, ?, ?, ?, ?)',
                    values
                )
            return True
        except Exception as e:
            logger.error(f"Error setting multiple items in cache: {e}")
            return False

    def clear(self):
        """Clear all cached data."""
        try:
            with self._get_connection() as conn:
                conn.execute('DELETE FROM cache')
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM cache')
                total_entries = cursor.fetchone()[0]
                
                cursor = conn.execute('SELECT model, COUNT(*) FROM cache GROUP BY model')
                model_counts = dict(cursor.fetchall())
                
                return {
                    'total_entries': total_entries,
                    'entries_by_model': model_counts
                }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'total_entries': 0, 'entries_by_model': {}}
