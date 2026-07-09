# ---------------------------------------------------------------------------
# LOCAL MODEL
# This script downloads the model weights to your machine and runs inference
# locally on your own hardware (CPU/GPU) - nothing is generated on a server.
#
# On the first run, the weights are downloaded and cached at:
#   C:\Users\user\.cache\huggingface\hub\
# Every run after that loads from that cache (no re-download, works offline).
#
# Because it runs locally, model size is limited by your RAM/CPU/GPU.
# Small models (like this one) are fine; very large models (e.g. GLM-5.2)
# won't run here - use the Hugging Face InferenceClient (remote) for those.
# ---------------------------------------------------------------------------

from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

messages = [
    {"role": "system", "content": "You are a geography expert who answers questions in detail. Every response should start with 'Fact:', followed by a newline"},
    {"role": "user", "content": "Tell me a cool geography fact"},
]

# Get 2 geography facts
results = pipe(messages, max_new_tokens=250, num_return_sequences=2)

for result in results:
    print(result["generated_text"][2]["content"], end="\n\n")
