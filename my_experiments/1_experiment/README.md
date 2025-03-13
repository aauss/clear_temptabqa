Date: 2025-03-12

# First experiments

## Model selection
During my first experiments, I tested:

- `microsoft/Phi-4-mini-instruct`
- `Warrieryes/timo-7b-hf`
- `meta-llama/Llama-3.1-8B-Instruct`

I focused on smaller models that would fit a single GPU. I could have picked `google/gemma-3-4b-it` instead of phi but a few benchmarks looked better for phi on QA. I could have picked `Qwen/Qwen2.5-7B-Instruct` or `mistralai/Ministral-8B-Instruct-2410` instead of Llama. But benchmarks for mistral did not look as good and Llama was more common in existing research. 

## Settings

I started with Llama and saw that it repeats random tables after giving a final answer. Therefore, I reduced the max_token_length to 70 to avoid this. I increased it for phi and Timo, since their output otherwise got truncated.

Otherwise, I kept the settings as the authors of CLEAR. I did not continue using the base model as indicated by them because instruction-tuned models are better at solving QA benchmarks usually.
