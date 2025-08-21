import os
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

if os.getenv("CI") != "true" and os.getenv("GITHUB_ACTIONS") != "true":
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass

from groq import Groq

GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct",
]

DEBUG_MODE = os.getenv("DEBUG_GROQ") == "1"


def extract_output(response, model_name: str):
    """Normalize Groq response so pytest and deepeval behave consistently."""
    if not response or not getattr(response, "choices", None):
        if DEBUG_MODE:
            print(f"[DEBUG] No choices in response for {model_name}: {response}")
        return None

    choice = response.choices[0]

    if getattr(choice, "message", None) and getattr(choice.message, "content", None):
        return choice.message.content

    if getattr(choice, "text", None):
        return choice.text

    if DEBUG_MODE:
        print(f"[DEBUG] Unexpected response schema for {model_name}: {choice}")

    return None


@pytest.mark.parametrize("model_name", GROQ_MODELS)
def test_case(model_name, request):
    # Skip integration tests when credentials are unavailable
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("Skipping: GROQ_API_KEY is not set")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Skipping: OPENAI_API_KEY is required for GEval")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    input_prompt = "Explain the role of Tier 1 suppliers in the automotive supply chain"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_prompt},
            ],
            temperature=0,
            seed=42,
            max_completion_tokens=512,
        )
    except Exception as e:
        pytest.skip(f"Groq chat.completions failed for {model_name}: {e}")

    actual_output = extract_output(response, model_name)
    if not actual_output:
        pytest.skip(f"Could not extract output for {model_name}, raw response: {response}")

    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.6,
    )

    test_case = LLMTestCase(
        input=input_prompt,
        actual_output=actual_output,
        expected_output=(
            "Tier 1 suppliers deliver complete systems or modules directly to automakers, "
            "such as braking systems, seats, or electronics, and typically manage sub-suppliers."
        ),
        retrieval_context=[
            "Tier 1 suppliers provide complex systems or assemblies directly to automotive OEMs.",
            "They source parts from lower-tier suppliers, ensure quality and compliance, and often co-develop designs with OEMs.",
        ],
    )

    results = assert_test(test_case, [correctness_metric])
