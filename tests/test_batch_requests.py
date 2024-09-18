import unittest
from unittest.mock import MagicMock, patch
import json
import os
import copy
import pytest
from openai import OpenAI
from simple_openai_requests import batch_requests as br

class TestBatchRequests(unittest.TestCase):
    
    def setUp(self):
        self.conversations = [
            {"index": i, "conversation": [{"role": "user", "content": f"Test message {i}"}]} for i in range(100)
        ]
        self.model_name = "gpt-3.5-turbo-0125"
        self.batch_dir = "batch_files"
        self.batch_run_name = "test_run"
        self.status_check_interval = 1  # Set a short interval for testing

    @pytest.mark.real
    def test_real_openai_requests(self):
        """Test real OpenAI requests without mocking."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set in environment")

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Ensure the output directory exists
        if not os.path.exists(self.batch_dir):
            os.makedirs(self.batch_dir)
        
        # Call make_batch_request with the real client
        results = br.make_batch_request(client, self.conversations[:10], self.model_name, self.batch_dir, self.batch_run_name, self.status_check_interval)
        
        # Verify the result length
        self.assertEqual(len(results), 10)
        
        # Check the content of the results
        for i, result in enumerate(results):
            self.assertEqual(result['conversation'][0]['content'], f"Test message {i}")
            self.assertIn('response', result)
            self.assertIn('choices', result['response'])
            self.assertIn('message', result['response']['choices'][0])
            self.assertIn('content', result['response']['choices'][0]['message'])

    @pytest.mark.mock
    @patch("batch_requests.OpenAI")
    def test_mock_requests(self, mock_openai_class):
        """Test batch processing with mocked OpenAI requests."""
        # Mock the OpenAI client instance
        mock_client = mock_openai_class.return_value
        mock_client.files.create.return_value = MagicMock(id="mock-file-id")
        mock_client.batches.create.return_value = MagicMock(id="mock-batch-id")
        
        # Simulate the batch retrieval progress
        mock_client.batches.retrieve.return_value.json.side_effect = [
            json.dumps({"status": "in_progress", "request_counts": {"total": 10, "completed": 1}}),
            json.dumps({"status": "in_progress", "request_counts": {"total": 10, "completed": 5}}),
            json.dumps({"status": "completed", "request_counts": {"total": 10, "completed": 10}, "output_file_id": "file-output123"})
        ]
        
        # Simulate the output file content (mocked output responses)
        mock_client.files.content.return_value.text = '\n'.join([json.dumps({
            "custom_id": f"request-{i}",
            "response": {"body": {"choices": [{"message": {"content": f"Response {i}"}}]}}
        }) for i in range(10)])
        
        # Use the mock client for batch processing
        results = br.make_batch_request(mock_client, self.conversations[:10], self.model_name, self.batch_dir, self.batch_run_name, self.status_check_interval)
        
        # Check that the results are correct
        self.assertEqual(len(results), 10)
        for i, result in enumerate(results):
            self.assertEqual(result['conversation'][0]['content'], f"Test message {i}")
            self.assertEqual(result['response']['choices'][0]['message']['content'], f"Response {i}")
        
        # Assert that the mock OpenAI API methods were called
        mock_client.files.create.assert_called_once()
        mock_client.batches.create.assert_called_once()
        self.assertEqual(mock_client.batches.retrieve.call_count, 3)
        mock_client.files.content.assert_called_once()

    @pytest.mark.mock
    @patch("batch_requests.OpenAI")
    def test_make_batch_request_multiple_batches(self, mock_openai_class):
        br.STATUS_CHECK_INTERVAL = 1
        br.MAX_REQUESTS_PER_BATCH = 50
        br.MAX_BATCH_FILE_SIZE_BYTES = 5000

        mock_client = mock_openai_class.return_value
        
        def mock_create_file(file, purpose):
            return MagicMock(id=f"mock-file-{file.name.split('_')[-2]}")
        mock_client.files.create.side_effect = mock_create_file

        def mock_batch_create(input_file_id, endpoint, completion_window):
            return MagicMock(id=f"mock-batch-{input_file_id}")
        mock_client.batches.create.side_effect = mock_batch_create
        
        def mock_batch_retrieve(batch_id):
            idx = int(batch_id.split('-')[-1])
            return type('MockObject', (), {'json': lambda self: json.dumps({
                "status": "completed",
                "request_counts": {"total": 10, "completed": 10},
                "output_file_id": f"file-output-{idx}"
            })})()

        mock_client.batches.retrieve.side_effect = mock_batch_retrieve

        def mock_file_content(file_id):
            idx = int(file_id.split('-')[-1])
            file_path = f"{self.batch_dir}/{self.batch_run_name}_batch_{idx}_request.jsonl"
            with open(file_path, 'r') as f:
                idx_list = [json.loads(line)['custom_id'].split('-')[-1] for line in f]
                response_text = '\n'.join([json.dumps({
                    "custom_id": f"request-{i}",
                    "response": {"body": {"choices": [{"message": {"content": f"Response {i}"}}]}}
                }) for i in idx_list])
                return MagicMock(text=response_text)

        mock_client.files.content.side_effect = mock_file_content

        results = br.make_batch_request_multiple_batches(mock_client, self.conversations, self.model_name, self.batch_dir, self.batch_run_name, self.status_check_interval)

        self.assertEqual(len(results), 100)
        for i, result in enumerate(results):
            self.assertEqual(result['conversation'][0]['content'], f"Test message {i}")
            self.assertEqual(result['response']['choices'][0]['message']['content'], f"Response {i}")

    @pytest.mark.mock
    def test_process_batch_output(self):
        batch = [{"index": i, "conversation": [{"role": "user", "content": f"Test message {i}"}]} for i in range(5)]
        output_file = os.path.join(self.batch_dir, "test_output.jsonl")
        with open(output_file, 'w') as f:
            for i in range(5):
                f.write(json.dumps({
                    "custom_id": f"request-{i}",
                    "response": {"body": {"choices": [{"message": {"content": f"Response {i}"}}]}}
                }) + '\n')

        results = br.process_batch_output(output_file, batch)

        self.assertEqual(len(results), 5)
        for i, result in enumerate(results):
            self.assertEqual(result['index'], i)
            self.assertEqual(result['conversation'][0]['content'], f"Test message {i}")
            self.assertEqual(result['response']['choices'][0]['message']['content'], f"Response {i}")

if __name__ == "__main__":
    # unittest.main()  

    # suite = unittest.TestSuite()
    # suite.addTest(TestBatchRequests('test_real_openai_requests'))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)

    pytest.main(["-v", "-m", "mock", "-k", "test_make_batch_request_multiple_batches", "test_batch_requests.py"])
    # pytest.main(["-v", "-m", "mock", "test_batch_requests.py"])
    # pytest.main(["-v", "-m", "mock"])

    # pytest.main(["-v", "-m", "real", "test_batch_requests.py"])
