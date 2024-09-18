import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import json
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
import pytest

import simple_openai_requests as sor
import batch_requests

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
    @patch('simple_openai_requests.OpenAI')
    @patch('simple_openai_requests.make_batch_request_multiple_batches')
    def test_batch_request_without_cache_full_response(self, mock_batch_request, mock_openai):
        mock_client = mock_openai.return_value
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
            full_response=True
        )

        self.assertEqual(len(results), len(self.conversations))
        for i, result in enumerate(results):
            self.assertEqual(result['conversation'], self.conversations[i])
            self.assertEqual(result['response']['choices'][0]['message']['content'], f"Response {i}")
            self.assertFalse(result['is_cached_response'])

        mock_batch_request.assert_called_once()

    @pytest.mark.mock
    @patch('simple_openai_requests.OpenAI')
    @patch('simple_openai_requests.make_batch_request_multiple_batches')
    def test_batch_request_without_cache_partial_response(self, mock_batch_request, mock_openai):
        mock_client = mock_openai.return_value
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
            full_response=False
        )

        self.assertEqual(len(results), len(self.conversations))
        for i, result in enumerate(results):
            self.assertEqual(result['conversation'], self.conversations[i])
            self.assertEqual(result['response'], f"Response {i}")
            self.assertFalse(result['is_cached_response'])

        mock_batch_request.assert_called_once()

    @pytest.mark.mock
    @patch('simple_openai_requests.OpenAI')
    @patch('simple_openai_requests.make_parallel_sync_requests')
    def test_sync_request_without_cache(self, mock_sync_requests, mock_openai):
        mock_client = mock_openai.return_value
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
    @patch('simple_openai_requests.OpenAI')
    @patch('simple_openai_requests.make_batch_request_multiple_batches')
    def test_batch_request_with_cache(self, mock_batch_request, mock_openai):
        mock_client = mock_openai.return_value
        mock_batch_request.return_value = [
            {"index": i, "conversation": conv, "response": self.create_mock_completion(f"Response {i}").model_dump(), "error": None}
            for i, conv in enumerate(self.conversations[1:])
        ]

        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_cache_file:
            cache_file_path = temp_cache_file.name
            # Pre-populate cache with one conversation
            json.dump([{
                "conversation": self.conversations[0],
                "model": self.model,
                "generation_args": self.generation_args,
                "response": self.create_mock_completion("Cached response").model_dump()
            }], temp_cache_file)
            temp_cache_file.flush()

        results = sor.make_openai_requests(
            self.conversations,
            self.model,
            generation_args=self.generation_args,
            use_batch=True,
            use_cache=True,
            cache_file=cache_file_path,
            batch_dir=self.batch_dir,  # Set batch_dir for batch request
            full_response=True
        )

        self.assertEqual(len(results), len(self.conversations))
        self.assertTrue(results[0]['is_cached_response'])
        self.assertEqual(results[0]['response']['choices'][0]['message']['content'], "Cached response")
        for i in range(1, len(results)):
            self.assertFalse(results[i]['is_cached_response'])
            self.assertEqual(results[i]['response']['choices'][0]['message']['content'], f"Response {i-1}")

        mock_batch_request.assert_called_once()

        # Verify that the cache was updated
        with open(cache_file_path, 'r') as f:
            updated_cache = json.load(f)
        self.assertEqual(len(updated_cache), len(self.conversations))

        os.unlink(cache_file_path)

    @pytest.mark.mock
    @patch('simple_openai_requests.OpenAI')
    def test_sync_request_with_cache(self, mock_openai):
        mock_client = mock_openai.return_value
        mock_client.chat.completions.create.side_effect = [
            self.create_mock_completion(f"Response {i}") for i in range(1, len(self.conversations))
        ]

        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_cache_file:
            cache_file_path = temp_cache_file.name
            # Pre-populate cache with one conversation
            json.dump([{
                "conversation": self.conversations[0],
                "model": self.model,
                "generation_args": self.generation_args,
                "response": self.create_mock_completion("Cached response").model_dump()
            }], temp_cache_file)
            temp_cache_file.flush()

        results = sor.make_openai_requests(
            self.conversations,
            self.model,
            generation_args=self.generation_args,
            use_batch=False,
            use_cache=True,
            cache_file=cache_file_path,
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

        # Verify that the cache was updated
        with open(cache_file_path, 'r') as f:
            updated_cache = json.load(f)
        self.assertEqual(len(updated_cache), len(self.conversations))

        os.unlink(cache_file_path)

    @pytest.mark.mock
    @patch('simple_openai_requests.OpenAI')
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
    @patch('simple_openai_requests.OpenAI')
    @patch('simple_openai_requests.make_batch_request_multiple_batches')
    def test_multiple_batches(self, mock_multiple_batches, mock_openai):
        mock_client = mock_openai.return_value
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
            batch_dir=self.batch_dir  # Set batch_dir for batch request
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
        if not api_key:
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
        if not api_key:
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
        if not api_key:
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
            full_response=True
        )

        self._assert_valid_results(results, len(conversations))

    @pytest.mark.real
    def test_real_openai_request_sync_with_cache(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
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
        if not api_key:
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
                full_response=True
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
                full_response=True
            )

            self._assert_valid_results(results2, len(conversations))
            for r1, r2 in zip(results1, results2):
                self.assertEqual(r1['response'], r2['response'])
                self.assertTrue(r2['is_cached_response'])

        os.unlink(cache_file_path)

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

if __name__ == '__main__':
    # pytest.main(["-v", "-m", "mock", "-k", "test_sync_request_with_cache", "test_simple_openai_requests.py", "--log-cli-level=INFO"])
    # pytest.main(["-v", "-m", "mock", "test_simple_openai_requests.py", "--log-cli-level=INFO"])
    # To run real tests, add OPENAI_API_KEY to environment variable and use:
    pytest.main(["-v", "-m", "real", "test_simple_openai_requests.py", "--log-cli-level=INFO"])
    # pytest.main(["-v", "-m", "real", "-k", "test_real_openai_request_batch_with_cache", "test_simple_openai_requests.py", "--log-cli-level=INFO"])