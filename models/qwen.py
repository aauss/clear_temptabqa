from accelerate.test_utils.testing import get_backend
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE, _, _ = get_backend()

model = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(
    model,
    device_map="auto",
    torch_dtype="auto",
)

def qwen(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_length = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
    )
    response_str = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response_str

def qwen_ct(messages: list[dict[str, str]]) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
