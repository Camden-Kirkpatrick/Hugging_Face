"""
Sentiment analysis using the manual (Auto-class) pipeline setup.

Instead of pipeline("sentiment-analysis", model="..."), this loads the model and
tokenizer explicitly with Auto classes, then passes them into pipeline(). Loading
them yourself gives you room to inspect or configure each piece.

Also prints the tokenized text to show what the model actually receives.

Model: distilbert-base-uncased-finetuned-sst-2-english (POSITIVE / NEGATIVE).
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Load the model and tokenizer manually (same checkpoint for both)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Build the pipeline from them
classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=0,
)

text = "This movie was fantastic!"

# Show how the tokenizer splits the text
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

result = classifier(text)
print("Result:", result)