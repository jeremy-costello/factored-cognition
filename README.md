# factored-cognition
Factored Cognition with LLMs.

This is a reimagining of Ought's [Factored Cognition Primer](https://primer.ought.org/).

## Requirements
The only requirement (besides Python) is [vLLM](https://docs.vllm.ai/en/latest/).

## Models
Supports any model in vLLM, including quantized models.

Adding a new model class in ```models.py```:
- The new model should be a subclass of the ```Model``` class.
  - Include vocab size, context length, and prompt templates.
- Subclass this new model class for specific instantiations of the model (e.g. sizes or quantizations).

See ```LLaMa2``` and ```LLaMa2_7B_Chat_AWQ``` in ```models.py```.

## What I've Implemented
- Question answering (with and without context)
- Debate (including a judge)
- Extracting title, authors, abstract, sections from PDFs

## To Do
- Long Texts
- Amplification
- Verifiers
- Tool Use
- Deduction
- Action Selection
