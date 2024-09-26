from simple_openai_requests import make_openai_requests
import json

def print_results(results):
    print('--- OUTPUT')
    for result in results:
        print(f"Index: {result.get('index', 'N/A')}")
        if isinstance(result['conversation'], list):
            print(f"Question: {result['conversation'][-1]['content']}")
        else:
            print(f"Question: {result['conversation']}")
        print(f"Answer: {result['response']}")
        print(f"Cached: {result['is_cached_response']}")
        print(f"Error: {result['error']}")
        print("-" * 50)

def run_example_1():
    print("Example 1: Simple string prompts")
    conversations = [
        "What is the capital of France?",
        "How does photosynthesis work?"
    ]

    results = make_openai_requests(
        conversations=conversations,
        model="gpt-3.5-turbo",
        use_batch=False,
        use_cache=False
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))
    # print_results(results)

def run_example_2():
    print("Example 2: Conversation format")
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
        model="gpt-4o-mini",
        use_batch=True,
        use_cache=False,
        generation_args={"max_tokens": 150}
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))

    # print_results(results)

def run_example_3():
    print("Example 3: Indexed conversation format")
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
        use_cache=False,
        max_workers=2
    )

    print_results(results)

if __name__ == "__main__":
    # run_example_1()
    # print("\n" + "=" * 70 + "\n")
    run_example_2()
    # print("\n" + "=" * 70 + "\n")
    # run_example_3()