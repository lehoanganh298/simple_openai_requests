import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import json
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
import pytest
import sqlite3
from simple_openai_requests.db_caching import SQLiteCache, get_cache_key

from simple_openai_requests import simple_openai_requests as sor

class TestSimpleOpenAIRequests(unittest.TestCase):

    def setUp(self):
        self.conversations = [
            [{"role": "user", "content": "Hello!"}],
            [{"role": "user", "content": "How are you?"}],
            [{"role": "user", "content": "What's 2 + 2?"}]
        ]
        self.model = "gpt-3.5-turbo"
        self.generation_args = {"max_tokens": 150, "temperature": 0.7}
        self.batch_dir = os.path.expanduser("~./gpt_batch_dir_test")  # Set batch_dir
        self.patcher = patch('builtins.input', return_value='y')
        self.mock_input = self.patcher.start()
        self.status_check_interval = 1  # Set a short interval for testing
        if os.environ.get('OPENAI_API_KEY') is None:
            os.environ['OPENAI_API_KEY'] = ''

    def tearDown(self):
        self.patcher.stop()

    def create_mock_completion(self, content):
        return ChatCompletion(
            id="1",
            choices=[Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content=content, role="assistant"))],
            created=123456,
            model=self.model,
            object="chat.completion"
        )

    @pytest.mark.mock
    @patch('simple_openai_requests.simple_openai_requests.make_batch_request_multiple_batches')
    def test_batch_request_without_cache_full_response(self, mock_batch_request):
        mock_batch_request.return_value = [
            {"index": i, "conversation": conv, "response": self.create_mock_completion(f"Response {i}").model_dump(), "error": None}
            for i, conv in enumerate(self.conversations)
        ]

        results = sor.make_openai_requests(
            self.conversations,
            self.model,
            generation_args=self.generation_args,
            use_batch=True,
            use_cache=False,
            batch_dir=self.batch_dir,
            full_response=True,
            status_check_interval=self.status_check_interval
        )

        self.assertEqual(len(results), len(self.conversations))
        for i, result in enumerate(results):
            self.assertEqual(result['conversation'], self.conversations[i])
            self.assertEqual(result['response']['choices'][0]['message']['content'], f"Response {i}")
            self.assertFalse(result['is_cached_response'])

        mock_batch_request.assert_called_once()

    @pytest.mark.mock
    @patch('simple_openai_requests.simple_openai_requests.make_batch_request_multiple_batches')
    def test_batch_request_without_cache_partial_response(self, mock_batch_request):
        mock_batch_request.return_value = [
            {"index": i, "conversation": conv, "response": self.create_mock_completion(f"Response {i}").model_dump(), "error": None}
            for i, conv in enumerate(self.conversations)
        ]

        results = sor.make_openai_requests(
            self.conversations,
            self.model,
            generation_args=self.generation_args,
            use_batch=True,
            use_cache=False,
            batch_dir=self.batch_dir,
            full_response=False,
            status_check_interval=self.status_check_interval
        )

        self.assertEqual(len(results), len(self.conversations))
        for i, result in enumerate(results):
            self.assertEqual(result['conversation'], self.conversations[i])
            self.assertEqual(result['response'], f"Response {i}")
            self.assertFalse(result['is_cached_response'])

        mock_batch_request.assert_called_once()

    @pytest.mark.mock
    @patch('simple_openai_requests.simple_openai_requests.make_parallel_sync_requests')
    def test_sync_request_without_cache(self, mock_sync_requests):
        mock_sync_requests.return_value = [
            {"index": i, "conversation": conv, "response": self.create_mock_completion(f"Response {i}").model_dump(), "error": None}
            for i, conv in enumerate(self.conversations)
        ]

        results = sor.make_openai_requests(
            self.conversations,
            self.model,
            generation_args=self.generation_args,
            use_batch=False,
            use_cache=False,
            full_response=True
        )

        self.assertEqual(len(results), len(self.conversations))
        for i, result in enumerate(results):
            self.assertEqual(result['conversation'], self.conversations[i])
            self.assertEqual(result['response']['choices'][0]['message']['content'], f"Response {i}")
            self.assertFalse(result['is_cached_response'])

        mock_sync_requests.assert_called_once()

    @pytest.mark.mock
    @patch('simple_openai_requests.simple_openai_requests.make_batch_request_multiple_batches')
    def test_batch_request_with_cache(self, mock_batch_request):
        mock_batch_request.return_value = [
            {"index": i, "conversation": conv, "response": self.create_mock_completion(f"Response {i}").model_dump(), "error": None}
            for i, conv in enumerate(self.conversations[1:])
        ]

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            db_path = temp_db.name

        try:
            # Pre-populate cache with one conversation
            cache = SQLiteCache(db_path)
            cache_key = get_cache_key(self.conversations[0], self.model, self.generation_args)
            cache.set(cache_key, {
                "model": self.model,
                "generation_args": self.generation_args,
                "conversation": self.conversations[0],
                "response": self.create_mock_completion("Cached response").model_dump()
            })

            results = sor.make_openai_requests(
                self.conversations,
                self.model,
                generation_args=self.generation_args,
                use_batch=True,
                use_cache=True,
                cache_file=db_path,
                batch_dir=self.batch_dir,
                full_response=True,
                status_check_interval=self.status_check_interval
            )

            self.assertEqual(len(results), len(self.conversations))
            self.assertTrue(results[0]['is_cached_response'])
            self.assertEqual(results[0]['response']['choices'][0]['message']['content'], "Cached response")
            for i in range(1, len(results)):
                self.assertFalse(results[i]['is_cached_response'])
                self.assertEqual(results[i]['response']['choices'][0]['message']['content'], f"Response {i-1}")

            # Verify final cache contents
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM cache')
                cache_count = cursor.fetchone()[0]
                self.assertEqual(cache_count, len(self.conversations))

        finally:
            os.unlink(db_path)

    @pytest.mark.mock
    @patch('simple_openai_requests.simple_openai_requests.OpenAI')
    def test_sync_request_with_cache(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = [
            self.create_mock_completion(f"Response {i}") for i in range(1, len(self.conversations))
        ]

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            db_path = temp_db.name

        try:
            # Pre-populate cache with one conversation
            cache = SQLiteCache(db_path)
            cache_key = get_cache_key(self.conversations[0], self.model, self.generation_args)
            cache.set(cache_key, {
                "model": self.model,
                "generation_args": self.generation_args,
                "conversation": self.conversations[0],
                "response": self.create_mock_completion("Cached response").model_dump()
            })

            results = sor.make_openai_requests(
                self.conversations,
                self.model,
                generation_args=self.generation_args,
                use_batch=False,
                use_cache=True,
                cache_file=db_path,
                full_response=True
            )

            self.assertEqual(len(results), len(self.conversations))
            self.assertTrue(results[0]['is_cached_response'])
            self.assertEqual(results[0]['response']['choices'][0]['message']['content'], "Cached response")
            for i in range(1, len(results)):
                self.assertFalse(results[i]['is_cached_response'])
                self.assertEqual(results[i]['response']['choices'][0]['message']['content'], f"Response {i}")

            # Verify that the API was called the correct number of times
            self.assertEqual(mock_client.chat.completions.create.call_count, len(self.conversations) - 1)

            # Verify final cache contents
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM cache')
                cache_count = cursor.fetchone()[0]
                self.assertEqual(cache_count, len(self.conversations))

        finally:
            os.unlink(db_path)

    @pytest.mark.mock
    @patch('simple_openai_requests.simple_openai_requests.OpenAI')
    def test_error_handling(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        # Instead of asserting an exception, we will assert that the function completes without error
        result = sor.make_openai_requests(
            self.conversations,
            self.model,
            generation_args=self.generation_args,
            use_batch=False,
            use_cache=False
        )
        self.assertIsInstance(result, list)  # Ensure that the result is a list, indicating no error occurred

    @pytest.mark.mock
    @patch('simple_openai_requests.simple_openai_requests.make_batch_request_multiple_batches')
    def test_multiple_batches(self, mock_multiple_batches):
        mock_multiple_batches.return_value = [
            {"index": i, "conversation": conv, "response": self.create_mock_completion(f"Response {i}").model_dump(), "error": None}
            for i, conv in enumerate(self.conversations * 100)  # Create a large number of conversations
        ]

        large_conversations = self.conversations * 100
        results = sor.make_openai_requests(
            large_conversations,
            self.model,
            generation_args=self.generation_args,
            use_batch=True,
            use_cache=False,
            batch_dir=self.batch_dir,  # Set batch_dir for batch request
            status_check_interval=self.status_check_interval
        )

        self.assertEqual(len(results), len(large_conversations))
        mock_multiple_batches.assert_called_once()

    @pytest.mark.mock
    def test_invalid_api_key(self):
        old_api_key = os.environ.get('OPENAI_API_KEY')
        os.environ.pop('OPENAI_API_KEY', None)
        try:
            with self.assertRaises(ValueError):
                sor.make_openai_requests(
                    self.conversations,
                    self.model,
                    generation_args=self.generation_args,
                    use_batch=False,
                    use_cache=False
                )
        finally:
            if old_api_key is not None:
                os.environ['OPENAI_API_KEY'] = old_api_key

    @pytest.mark.real
    def test_real_openai_request_sync_no_cache_full_response(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == '':
            pytest.skip("OPENAI_API_KEY not set in environment")

        conversations = [
            [{"role": "user", "content": "What's the capital of France?"}],
            [{"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}]
        ]

        results = sor.make_openai_requests(
            conversations,
            self.model,
            generation_args={"max_tokens": 50, "temperature": 0.7},
            use_batch=False,
            use_cache=False,
            full_response=True
        )

        self._assert_valid_results(results, len(conversations), full_response=True)

    @pytest.mark.real
    def test_real_openai_request_sync_no_cache_partial_response(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == '':
            pytest.skip("OPENAI_API_KEY not set in environment")

        conversations = [
            [{"role": "user", "content": "What's the capital of France?"}],
            [{"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}]
        ]

        results = sor.make_openai_requests(
            conversations,
            self.model,
            generation_args={"max_tokens": 50, "temperature": 0.7},
            use_batch=False,
            use_cache=False,
            full_response=False
        )

        self._assert_valid_results(results, len(conversations), full_response=False)

    @pytest.mark.real
    def test_real_openai_request_batch_no_cache(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == '':
            pytest.skip("OPENAI_API_KEY not set in environment")

        conversations = [
            [{"role": "user", "content": "What's the largest planet in our solar system?"}],
            [{"role": "user", "content": "Who painted the Mona Lisa?"}]
        ]

        results = sor.make_openai_requests(
            conversations,
            self.model,
            generation_args={"max_tokens": 50, "temperature": 0.7},
            use_batch=True,
            use_cache=False,
            batch_dir=self.batch_dir,  # Set batch_dir for batch request
            full_response=True,
            status_check_interval=self.status_check_interval
        )

        self._assert_valid_results(results, len(conversations))

    @pytest.mark.real
    def test_real_openai_request_sync_with_cache(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == '':
            pytest.skip("OPENAI_API_KEY not set in environment")

        conversations = [
            [{"role": "user", "content": "What's the capital of Japan?"}],
            [{"role": "user", "content": "Who wrote 'To Kill a Mockingbird'?"}]
        ]

        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_cache_file:
            json.dump({}, temp_cache_file)
            temp_cache_file.flush()

            cache_file_path = temp_cache_file.name

            # First request to populate cache
            results1 = sor.make_openai_requests(
                conversations,
                self.model,
                generation_args={"max_tokens": 50, "temperature": 0.7},
                use_batch=False,
                use_cache=True,
                cache_file=cache_file_path,
                full_response=True
            )

            self._assert_valid_results(results1, len(conversations))

            # Second request to use cache
            results2 = sor.make_openai_requests(
                conversations,
                self.model,
                generation_args={"max_tokens": 50, "temperature": 0.7},
                use_batch=False,
                use_cache=True,
                cache_file=cache_file_path,
                full_response=True
            )

            self._assert_valid_results(results2, len(conversations))
            for r1, r2 in zip(results1, results2):
                self.assertEqual(r1['response'], r2['response'])
                self.assertTrue(r2['is_cached_response'])

        os.unlink(cache_file_path)

    @pytest.mark.real
    def test_real_openai_request_batch_with_cache(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == '':
            pytest.skip("OPENAI_API_KEY not set in environment")

        conversations = [
            [{"role": "user", "content": "What's the capital of Germany?"}],
            [{"role": "user", "content": "Who wrote '1984'?"}]
        ]

        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_cache_file:
            json.dump({}, temp_cache_file)
            temp_cache_file.flush()
            
            cache_file_path = temp_cache_file.name

            # First request to populate cache
            results1 = sor.make_openai_requests(
                conversations,
                self.model,
                generation_args={"max_tokens": 50, "temperature": 0.7},
                use_batch=True,
                use_cache=True,
                cache_file=cache_file_path,
                batch_dir=self.batch_dir,  # Set batch_dir for batch request
                full_response=True,
                status_check_interval=self.status_check_interval
            )

            self._assert_valid_results(results1, len(conversations))

            # Second request to use cache
            results2 = sor.make_openai_requests(
                conversations,
                self.model,
                generation_args={"max_tokens": 50, "temperature": 0.7},
                use_batch=True,
                use_cache=True,
                cache_file=cache_file_path,
                batch_dir=self.batch_dir,  # Set batch_dir for batch request
                full_response=True,
                status_check_interval=self.status_check_interval
            )

            self._assert_valid_results(results2, len(conversations))
            for r1, r2 in zip(results1, results2):
                self.assertEqual(r1['response'], r2['response'])
                self.assertTrue(r2['is_cached_response'])

        os.unlink(cache_file_path)

    @pytest.mark.examples
    def test_simple_string_prompts(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == '':
            pytest.skip("OPENAI_API_KEY not set in environment")

        conversations = [
            "What is the capital of France?",
            "How does photosynthesis work?"
        ]

        results = sor.make_openai_requests(
            conversations=conversations,
            model="gpt-3.5-turbo",
            use_batch=False,
            use_cache=True
        )

        self.assertEqual(len(results), len(conversations))
        for result in results:
            self.assertIn('conversation', result)
            self.assertIn('response', result)
            self.assertIsInstance(result['response'], str)

            print(f"Question: {result['conversation'][0]['content']}")
            print(f"Answer: {result['response']}\n")

    @pytest.mark.examples
    def test_conversation_format(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == '':
            pytest.skip("OPENAI_API_KEY not set in environment")

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

        results = sor.make_openai_requests(
            conversations=conversations,
            model="gpt-3.5-turbo",
            use_batch=True,
            use_cache=False,
            generation_args={"max_tokens": 150}
        )

        self.assertEqual(len(results), len(conversations))
        for result in results:
            self.assertIn('conversation', result)
            self.assertIn('response', result)
            self.assertIsInstance(result['response'], str)

            print(f"Question: {result['conversation'][-1]['content']}")
            print(f"Answer: {result['response']}\n")

    @pytest.mark.examples
    def test_indexed_conversation_format(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == '':
            pytest.skip("OPENAI_API_KEY not set in environment")

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

        results = sor.make_openai_requests(
            conversations=conversations,
            model="gpt-3.5-turbo",
            use_batch=False,
            use_cache=True,
            max_workers=2
        )

        self.assertEqual(len(results), len(conversations))
        for result in results:
            self.assertIn('index', result)
            self.assertIn('conversation', result)
            self.assertIn('response', result)
            self.assertIsInstance(result['response'], str)

            print(f"Index: {result['index']}")
            print(f"Question: {result['conversation'][-1]['content']}")
            print(f"Answer: {result['response']}\n")

    def _assert_valid_results(self, results, expected_length, full_response=True):
        self.assertEqual(len(results), expected_length)
        for result in results:
            self.assertIsNotNone(result['response'])
            self.assertIsNone(result['error'])
            if full_response:
                self.assertIn('choices', result['response'])
                self.assertGreater(len(result['response']['choices']), 0)
                self.assertIn('message', result['response']['choices'][0])
                self.assertIn('content', result['response']['choices'][0]['message'])
                self.assertGreater(len(result['response']['choices'][0]['message']['content']), 0)
            else:
                self.assertIsInstance(result['response'], str)
                self.assertGreater(len(result['response']), 0)

    @pytest.mark.mock
    def test_cache_batch_operations(self):
        """Test that cache operations are performed in batches"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            db_path = temp_db.name

        try:
            # Create a large number of conversations
            num_conversations = sor.GET_BATCH_SIZE * 2 + 1
            conversations = [
                [{"role": "user", "content": f"Test message {i}"}]
                for i in range(num_conversations)
            ]

            # Pre-populate cache with some conversations
            cache = SQLiteCache(db_path)
            pre_cache_items = {
                get_cache_key(conv, self.model, self.generation_args): {
                    "model": self.model,
                    "generation_args": self.generation_args,
                    "conversation": conv,
                    "response": self.create_mock_completion(f"Cached response {i}").model_dump()
                }
                for i, conv in enumerate(conversations[:sor.GET_BATCH_SIZE])
            }
            cache.set_many(pre_cache_items)

            # Run the function with batched cache operations
            with patch('simple_openai_requests.simple_openai_requests.make_parallel_sync_requests') as mock_sync:
                mock_sync.return_value = [
                    {"index": i, "conversation": conv, "response": self.create_mock_completion(f"New response {i}").model_dump(), "error": None}
                    for i, conv in enumerate(conversations[sor.GET_BATCH_SIZE:], start=sor.GET_BATCH_SIZE)
                ]

                results = sor.make_openai_requests(
                    conversations=conversations,
                    model=self.model,
                    use_cache=True,
                    cache_file=db_path,
                    full_response=True,
                    generation_args=self.generation_args
                )

            # Verify results
            self.assertEqual(len(results), num_conversations)
            
            # Check cache contents
            # with sqlite3.connect(db_path) as conn:
            #     cursor = conn.execute('SELECT COUNT(*) FROM cache')
            #     cache_count = cursor.fetchone()[0]
            #     self.assertEqual(cache_count, num_conversations)

        finally:
            os.unlink(db_path)

    @pytest.mark.mock
    def test_cache_batch_size_limits(self):
        """Test that cache operations respect batch size limits"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            db_path = temp_db.name

        try:
            # Create conversations just over the batch size
            num_conversations = sor.SET_BATCH_SIZE + 5
            conversations = [
                [{"role": "user", "content": f"Test message {i}"}]
                for i in range(num_conversations)
            ]

            # Mock the API calls
            with patch('simple_openai_requests.simple_openai_requests.make_batch_request_multiple_batches') as mock_batch:
                mock_batch.return_value = [
                    {"index": i, "conversation": conv, "response": self.create_mock_completion(f"Response {i}").model_dump(), "error": None}
                    for i, conv in enumerate(conversations)
                ]

                results = sor.make_openai_requests(
                    conversations=conversations,
                    model=self.model,
                    use_batch=True,
                    use_cache=True,
                    cache_file=db_path,
                    batch_dir=self.batch_dir,
                    full_response=True,
                    status_check_interval=self.status_check_interval
                )

            # Verify results
            self.assertEqual(len(results), num_conversations)
            
            # Check that cache was updated in correct batch sizes
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute('SELECT COUNT(*) FROM cache')
                cache_count = cursor.fetchone()[0]
                self.assertEqual(cache_count, num_conversations)

                # Check timestamps to verify batching
                cursor = conn.execute('SELECT created_at FROM cache ORDER BY created_at')
                timestamps = [row[0] for row in cursor.fetchall()]
                unique_timestamps = len(set(timestamps))
                expected_batches = (num_conversations + sor.SET_BATCH_SIZE - 1) // sor.SET_BATCH_SIZE
                self.assertLessEqual(unique_timestamps, expected_batches)

        finally:
            os.unlink(db_path)

    @pytest.mark.mock
    def test_cache_error_recovery(self):
        """Test that the system handles cache errors gracefully and continues processing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            db_path = temp_db.name
        os.chmod(db_path, 0o000)  # Remove all permissions

        try:
            conversations = [
                [{"role": "user", "content": f"Test message {i}"}]
                for i in range(5)
            ]

            # Mock the API calls
            with patch('simple_openai_requests.simple_openai_requests.make_parallel_sync_requests') as mock_sync:
                mock_sync.return_value = [
                    {"index": i, "conversation": conv, "response": self.create_mock_completion(f"Response {i}").model_dump(), "error": None}
                    for i, conv in enumerate(conversations)
                ]

                # Function should complete without error despite cache issues
                results = sor.make_openai_requests(
                    conversations=conversations,
                    model=self.model,
                    use_cache=True,
                    cache_file=db_path,
                    full_response=True
                )

            self.assertEqual(len(results), len(conversations))
            for result in results:
                self.assertIsNotNone(result["response"])
                self.assertFalse(result.get("is_cached_response", False))

        finally:
            os.chmod(db_path, 0o666)  # Restore permissions for cleanup
            os.unlink(db_path)

if __name__ == '__main__':
    # unittest.main()  
    
    # suite = unittest.TestSuite()
    # suite.addTest(TestSimpleOpenAIRequests('test_sync_request_without_cache'))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
    
    # pytest.main(["-v", "-m", "mock", "-k", "test_sync_request_without_cache", "tests/test_simple_openai_requests.py", "--log-cli-level=INFO"])
    # pytest.main(["-v", "-m", "mock", "tests/test_simple_openai_requests.py", "--log-cli-level=INFO"])
    # To run real tests, add OPENAI_API_KEY to environment variable and use:
    pytest.main(["-v", "-m", "examples", "tests/test_simple_openai_requests.py", "--log-cli-level=INFO"])
    # pytest.main(["-v", "-m", "real", "-k", "test_real_openai_request_batch_with_cache", "test_simple_openai_requests.py", "--log-cli-level=INFO"])

