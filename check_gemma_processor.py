import transformers
from transformers import AutoProcessor
print("Loading processor...")
proc = AutoProcessor.from_pretrained("mlx-community/gemma-4-e4b-it-4bit")
print("Processor:", type(proc))
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "audio"}, {"type": "text", "text": "Describe."}]}]
print("Applying template...")
print(proc.apply_chat_template(messages, add_generation_prompt=True))
