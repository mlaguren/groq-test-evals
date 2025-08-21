# Groq Test Evals

This repo demonstrates how to run different evaluation frameworks against models hosted on [Groq](https://groq.com).

# Set Up

## Requirements
* Python3*

## API Keys
* Groq API Key
* Deep Eval API Key
* OpenAI API Key

## Setting up python virtual environment

1. Create virtual environment with latest system dependencies

```python3 -m venv --upgrade-deps venv-groq```

2. Start python virtual environment

```source venv-groq/bin/activate```

## Installing python modules

```pip install -r requirements.txt```

# Running Test

## Available Models on Groq

Models available on (GroqCloud)[https://console.groq.com/docs/models]

## DeepEval

There are two approaches to running the example test case which would display the results either locally or on [Confident AI](https://confident-ai.com)

### Locally

```pytest tests/test_deep_evals.py```

### Confident AI Dashboard

Log in to Confident AI 
```deepeval login```

Run test
``` deepeval test run tests/test_deep_evals.py```

