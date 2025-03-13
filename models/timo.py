from accelerate.test_utils.testing import get_backend
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv("../../.env")
DEVICE, _, _ = get_backend()

model = "Warrieryes/timo-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(
    model,
    use_fast=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model,
    torch_dtype="auto",
).to(DEVICE)

def timo(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_length = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id,
    )
    response_str = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    return response_str

def timo_ct(prompt: str) -> str:
    template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{query}\n\n### Response:"""

    input = template.format(query=prompt)
    inputs = tokenizer(input, return_tensors="pt").to(DEVICE)
    input_length = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
    )
    response_str = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    return response_str
