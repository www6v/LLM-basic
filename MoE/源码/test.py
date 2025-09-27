from transformers import Qwen3ForCausalLM, AutoTokenizer
model_name = "D:/AI/models_datasets/qwen3-8b"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen3ForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cpu"
)

# prepare the model input
prompt = "Give me a short introduction to large language models."
messages = [
    {"role": "user", "content": prompt}
    
]
text_1 = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)

text_2 = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text_1], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# the result will begin with thinking content in <think></think> tags, followed by the actual response
print(tokenizer.decode(output_ids, skip_special_tokens=True))