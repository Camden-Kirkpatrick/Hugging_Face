# ---------------------------------------------------------------------------
# REMOTE MODEL
# This script runs inference on Hugging Face's servers, NOT your machine.
# Nothing is downloaded; you send messages and get the response back.
# Requires an HF token with "Make calls to Inference Providers" permission,
# read from the HF_TOKEN environment variable.
# ---------------------------------------------------------------------------

from huggingface_hub import InferenceClient

client = InferenceClient()  # reads HF_TOKEN from your environment

messages = [
    {"role": "system", "content": "You are a geography expert who answers questions in detail. Every response should start with 'Fact:', followed by a newline"},
    {"role": "user", "content": "Tell me a cool geography fact"},
]

out = client.chat_completion(
    messages,
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_tokens=250,
)

print(out.choices[0].message.content)