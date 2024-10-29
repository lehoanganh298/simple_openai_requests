import pytest
import unittest
from unittest.mock import patch, MagicMock, call
import concurrent.futures
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
import os
import json
import tempfile
from openai import OpenAI
import sqlite3
from simple_openai_requests.db_caching import SQLiteCache, get_cache_key

# Import the functions to be tested
from simple_openai_requests import synchronous_requests as sr
from simple_openai_requests.synchronous_requests import make_api_call, make_parallel_sync_requests, CACHE_SAVE_INTERVAL

class TestParallelSyncRequests(unittest.TestCase):

    def setUp(self):
        self.model = "gpt-3.5-turbo"
        self.generation_args = {"max_tokens": 150, "temperature": 0.7}
        self.max_workers = 3
        self.max_retries = 3
        self.retry_delay = 1
        self.conversations = [
            [{"role": "user", "content": "Hello!"}],
            [{"role": "user", "content": "How are you?"}],
            [{"role": "user", "content": "What's 2 + 2?"}]
        ]

    @pytest.mark.mock
    @patch('simple_openai_requests.synchronous_requests.OpenAI')
    def test_successful_api_call(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_completion = ChatCompletion(
            created=123456,
            id="1", 
            choices=[Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content="Hello! How can I assist you today?", role="assistant"))],
            model=self.model,
            object="chat.completion"
        )
        mock_client.chat.completions.create.return_value = mock_completion

        result = make_api_call(mock_client, self.conversations[0], self.model, self.generation_args, self.max_retries, self.retry_delay)

        self.assertIsNone(result["error"])
        self.assertEqual(result["response"]["choices"][0]["message"]["content"], "Hello! How can I assist you today?")

    @pytest.mark.mock
    @patch('simple_openai_requests.synchronous_requests.OpenAI')
    @patch('simple_openai_requests.synchronous_requests.time.sleep', return_value=None)
    def test_rate_limit_with_retry(self, mock_sleep, mock_openai):
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = [
            Exception("Rate limit exceeded"),
            Exception("Rate limit exceeded"),
            ChatCompletion(
                created=123456,
                id="1", 
                choices=[Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content="Hello! How can I assist you today?", role="assistant"))],
                model=self.model,
                object="chat.completion"
            )
        ]
        mock_sleep.return_value = ['fadsfasdfaffsd']
        result = sr.make_api_call(mock_client, self.conversations[0], self.model, self.generation_args, self.max_retries, self.retry_delay)

        self.assertIsNone(result["error"])
        self.assertEqual(result["response"]["choices"][0]["message"]["content"], "Hello! How can I assist you today?")
        self.assertEqual(mock_sleep.call_count, 2)

    @pytest.mark.mock
    @patch('simple_openai_requests.synchronous_requests.OpenAI')
    @patch('simple_openai_requests.synchronous_requests.time.sleep', return_value=None)
    def test_rate_limit_max_retries_exceeded(self, mock_sleep, mock_openai):
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")

        result = make_api_call(mock_client, self.conversations[0], self.model, self.generation_args, self.max_retries, self.retry_delay)

        self.assertEqual(result["error"], "Rate limit exceeded")
        self.assertIsNone(result["response"])
        self.assertEqual(mock_sleep.call_count, self.max_retries - 1)

    @pytest.mark.mock
    @patch('simple_openai_requests.synchronous_requests.OpenAI')
    def test_other_api_error(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = Exception("API error")

        result = make_api_call(mock_client, self.conversations[0], self.model, self.generation_args, self.max_retries, self.retry_delay)

        self.assertEqual(result["error"], "API error")
        self.assertIsNone(result["response"])

    @pytest.mark.mock
    @patch('simple_openai_requests.synchronous_requests.OpenAI')
    def test_parallel_requests_all_successful(self, mock_openai):
        mock_client = mock_openai.return_value
        
        mock_completion = ChatCompletion(
            created=123456,
            id="1", 
            choices=[Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content="Response", role="assistant"))],
            model=self.model,
            object="chat.completion"
        )
        mock_client.chat.completions.create.return_value = mock_completion

        results = make_parallel_sync_requests(mock_client, self.conversations, self.model, self.generation_args, self.max_workers, self.max_retries, self.retry_delay)

        self.assertEqual(len(results), len(self.conversations))
        for idx, result in enumerate(results):
            self.assertEqual(result['conversation'], self.conversations[idx])
        for result in results:
            self.assertIsNone(result["error"])
            self.assertEqual(result["response"]["choices"][0]["message"]["content"], "Response")

    @pytest.mark.mock
    @patch('simple_openai_requests.synchronous_requests.OpenAI')
    @patch('simple_openai_requests.synchronous_requests.save_cache')
    def test_parallel_requests_mixed_results(self, mock_make_api_call, mock_openai):
        mock_client = mock_openai.return_value
        mock_completion = ChatCompletion(
            created=123456,
            id="1", 
            choices=[Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content="Response", role="assistant"))],
            model=self.model,
            object="chat.completion"
        )

        mock_client.chat.completions.create.side_effect = [
            mock_completion,
            Exception("Rate limit exceeded"),
            Exception("API error")] + [Exception("Rate limit exceeded")] * self.max_retries
        
        mock_make_api_call.return_value = 'fasdfafdfdfdfdfdf'
        results = make_parallel_sync_requests(mock_client, self.conversations, self.model, self.generation_args, self.max_workers, self.max_retries, self.retry_delay)

        self.assertEqual(len(results), len(self.conversations))
        self.assertIsNone(results[0]["error"])
        self.assertEqual(results[0]["response"]["choices"][0]["message"]["content"], "Response")
        self.assertEqual(results[1]["error"], "Rate limit exceeded")
        self.assertEqual(results[2]["error"], "API error")


    @pytest.mark.mock
    @patch('simple_openai_requests.synchronous_requests.OpenAI')
    def test_cache_saving(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_completion = ChatCompletion(
            created=123456,
            id="1", 
            choices=[Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content="Cached response", role="assistant"))],
            model=self.model,
            object="chat.completion"
        )
        mock_client.chat.completions.create.return_value = mock_completion

        # Create a temporary cache file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_cache_file:
            cache_file_path = temp_cache_file.name

        # Run the function with caching
        make_parallel_sync_requests(mock_client, self.conversations, self.model, self.generation_args, 
                                    self.max_workers, self.max_retries, self.retry_delay, use_cache=True,
                                    cache={}, cache_file=cache_file_path)

        # Load the cache from the file
        saved_cache = load_cache(cache_file_path)

        # Clean up the temporary file
        os.unlink(cache_file_path)

        # Assertions
        self.assertEqual(len(saved_cache), len(self.conversations))
        for key, item in saved_cache.items():
            self.assertIn('conversation', item)
            self.assertIn('model', item)
            self.assertIn('generation_args', item)
            self.assertIn('response', item)
            self.assertEqual(item['response']['choices'][0]['message']['content'], "Cached response")

    @pytest.mark.mock
    @patch('simple_openai_requests.synchronous_requests.OpenAI')
    @patch('simple_openai_requests.synchronous_requests.save_cache')
    def test_cache_update_and_save_interval(self, mock_save_cache, mock_openai):
        """Test that cache updates are saved in batches according to CACHE_SAVE_INTERVAL"""
        mock_client = mock_openai.return_value
        mock_completion = ChatCompletion(
            created=123456,
            id="1", 
            choices=[Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content="Cached response", role="assistant"))],
            model=self.model,
            object="chat.completion"
        )
        mock_client.chat.completions.create.return_value = mock_completion

        # Create more conversations than CACHE_SAVE_INTERVAL
        num_conversations = CACHE_SAVE_INTERVAL * 2 + 1
        conversations = [
            [{"role": "user", "content": f"Test message {i}"}]
            for i in range(num_conversations)
        ]

        # Create a temporary SQLite database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            db_path = temp_db.name

        try:
            cache = SQLiteCache(db_path)
            
            # Run the function with caching
            results = make_parallel_sync_requests(
                mock_client, conversations, self.model, 
                self.generation_args, self.max_workers, 
                self.max_retries, self.retry_delay,
                use_cache=True, cache=cache, cache_file=db_path
            )

            # Verify results
            self.assertEqual(len(results), num_conversations)
            
            # Check cache contents and verify batching
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM cache')
                cache_count = cursor.fetchone()[0]
                self.assertEqual(cache_count, num_conversations)

                # Check timestamps to verify batching
                cursor = conn.execute('SELECT created_at FROM cache ORDER BY created_at')
                timestamps = [row[0] for row in cursor.fetchall()]
                unique_timestamps = len(set(timestamps))
                expected_batches = (num_conversations // CACHE_SAVE_INTERVAL) + 1
                self.assertLessEqual(unique_timestamps, expected_batches)

        finally:
            os.unlink(db_path)

    @pytest.mark.mock
    @patch('simple_openai_requests.synchronous_requests.OpenAI')
    def test_cache_batch_saving(self, mock_openai):
        """Test that cache updates are saved in batches according to CACHE_SAVE_INTERVAL"""
        mock_client = mock_openai.return_value
        num_conversations = sr.CACHE_SAVE_INTERVAL * 2 + 1  # Create more conversations than CACHE_SAVE_INTERVAL
        
        # Create mock completion response
        mock_completion = ChatCompletion(
            created=123456,
            id="1", 
            choices=[Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content="Test response", role="assistant"))],
            model=self.model,
            object="chat.completion"
        )
        mock_client.chat.completions.create.return_value = mock_completion

        # Create a temporary SQLite database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            db_path = temp_db.name

        cache = SQLiteCache(db_path)
        conversations = [
            [{"role": "user", "content": f"Test message {i}"}]
            for i in range(num_conversations)
        ]

        # Run the function with caching
        results = make_parallel_sync_requests(
            mock_client, conversations, self.model, 
            self.generation_args, self.max_workers, 
            self.max_retries, self.retry_delay,
            use_cache=True, cache=cache
        )

        # Check results
        self.assertEqual(len(results), num_conversations)
        
        # Verify cache contents
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM cache')
            cache_count = cursor.fetchone()[0]
            self.assertEqual(cache_count, num_conversations)

        # Clean up
        os.unlink(db_path)

    # @pytest.mark.mock
    # @patch('simple_openai_requests.synchronous_requests.OpenAI')
    # def test_cache_error_handling(self, mock_openai):
    #     """Test that the system handles cache errors gracefully"""
    #     mock_client = mock_openai.return_value
    #     mock_completion = ChatCompletion(
    #         created=123456,
    #         id="1", 
    #         choices=[Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content="Test response", role="assistant"))],
    #         model=self.model,
    #         object="chat.completion"
    #     )
    #     mock_client.chat.completions.create.return_value = mock_completion

    #     # Create a temporary file with invalid permissions
    #     with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
    #         db_path = temp_db.name
    #     os.chmod(db_path, 0o000)  # Remove all permissions

    #     try:
    #         cache = SQLiteCache(db_path)
    #         # Function should complete without error despite cache issues
    #         results = make_parallel_sync_requests(
    #             mock_client, self.conversations, self.model,
    #             self.generation_args, self.max_workers,
    #             self.max_retries, self.retry_delay,
    #             use_cache=True, cache=cache
    #         )

    #         self.assertEqual(len(results), len(self.conversations))
    #         for result in results:
    #             self.assertIsNotNone(result["response"])
    #     finally:
    #         os.chmod(db_path, 0o666)  # Restore permissions for cleanup
    #         os.unlink(db_path)

    @pytest.mark.real
    def test_parallel_requests_all_successful_real_requests(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == '':
            pytest.skip("OPENAI_API_KEY not set in environment")

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        results = make_parallel_sync_requests(client, self.conversations, self.model, self.generation_args, self.max_workers, self.max_retries, self.retry_delay)

        self.assertEqual(len(results), len(self.conversations))
        for idx, result in enumerate(results):
            self.assertEqual(result['conversation'], self.conversations[idx])
        
        print(results)
        for result in results:
            self.assertIsNone(result["error"])
            self.assertIsNotNone(result["response"])

if __name__ == '__main__':
    # unittest.main()  
    
    # suite = unittest.TestSuite()
    # suite.addTest(TestParallelSyncRequests('test_parallel_requests_all_successful_real_requests'))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)

    pytest.main(["-v", "-k", "test_parallel_requests_mixed_results", "tests/test_synchronous_requests.py"])
    # pytest.main(["-v", "-m", "mock", "test_simple_openai_requests.synchronous_requests.py", "--log-cli-level=INFO"])
    # pytest.main(["-v", "-m", "mock"])

    # pytest.main(["-v", "-m", "real", "test_simple_openai_requests.synchronous_requests.py"])

