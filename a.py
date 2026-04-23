from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")


text = "what's the weather today in LA?"
encoded_with_special = tokenizer.encode(text, add_special_tokens=True)
encoded_with_special2 = tokenizer.encode(text, add_special_tokens=False)
print("---")
print(encoded_with_special)
print(encoded_with_special2)
print("---")
decoded_with_special = tokenizer.decode(encoded_with_special, skip_special_tokens=False)
print("Decoded with special tokens:", decoded_with_special)
decoded_with_special = tokenizer.decode(encoded_with_special, skip_special_tokens=True)
print("Decoded with special tokens2:", decoded_with_special)


chat = [
    {"role": "system", "content": "You are friendly agent."},
    {"role": "user", "content": "What's the weather today in LA?"},
    {"role": "assistant", "content": "The weather in LA is sunny with a high of 75°F."},
    {"role": "user", "content": "Will it rain tomorrow?"},
    {"role": "assistant", "content": "No, it's expected to be clear with a low of 58°F."},
]

encoded_chat = tokenizer.apply_chat_template(chat, tokenize=False)
print(f"Encoded chat:\n{encoded_chat}")