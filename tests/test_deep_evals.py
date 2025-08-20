import os
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval, ContextualRelevancyMetric, BiasMetric
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
    "moonshotai/kimi-k2-instruct"
]

@pytest.mark.parametrize("model_name", GROQ_MODELS)
def test_case(model_name):

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    input_prompt = "Explain the role of Tier 1 suppliers in the automotive supply chain"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_prompt},
        ],
    )

    actual_output = response.choices[0].message.content
    
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.7
    )

    faithfulness_metric = GEval(
        name="Faithfulness",
        criteria="Is the extracted data faithful to the provided input document without hallucination?"
    )

    contextual_relevancy = ContextualRelevancyMetric(
        threshold=0.7,
        include_reason=True
    )

    test_case = LLMTestCase(
        input="Explain the role of Tier 1 suppliers in the automotive supply chain.",
        actual_output=actual_output,
        expected_output="Tier 1 suppliers deliver complete systems or modules directly to automakers, such as braking systems, seats, or electronics, and typically manage sub-suppliers for individual components.",
        retrieval_context=["Tier 1 suppliers provide complex systems or assemblies (e.g., brakes, electronics, seats) directly to automotive OEMs.","They source parts from lower-tier suppliers, ensure quality and compliance, and often co-develop designs with OEMs."]
    )
    assert_test(test_case, [correctness_metric, contextual_relevancy, faithfulness_metric])