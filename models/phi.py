import os
from pathlib import Path

from accelerate.test_utils.testing import get_backend
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

file_path = Path(__file__).parent.resolve()
load_dotenv(str(file_path / "../../.env"), verbose=True)
ACCESS_TOKEN = os.environ["HF_TOKEN"]
DEVICE, _, _ = get_backend()

# model = "microsoft/Phi-4-mini-instruct"
model = "microsoft/phi-4"

tokenizer = AutoTokenizer.from_pretrained(
    model,
    use_fast=True,
    token=ACCESS_TOKEN,
)
model = AutoModelForCausalLM.from_pretrained(
    model,
    device_map="auto",
    torch_dtype="auto",
    token=ACCESS_TOKEN,
)


def phi(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_length = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
    )
    response_str = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response_str


def phi_ct(messages: list[dict[str, str]]) -> str:
    chat = tokenizer.apply_chat_template(
        messages,
        tokenizer=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(DEVICE)
    chat_length = chat["input_ids"].shape[1]
    outputs = model.generate(
        **chat,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
    )
    response_str = tokenizer.decode(outputs[0][chat_length:], skip_special_tokens=True)
    return response_str
