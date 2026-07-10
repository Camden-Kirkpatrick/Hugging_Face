"""
Explore the IMDB movie-review dataset from the Hugging Face Hub.

Loads "stanfordnlp/imdb" (50k reviews labeled for sentiment) and inspects it two ways:
  1. Full DatasetDict  -> load_dataset("stanfordnlp/imdb")
     Gives all splits (train / test / unsupervised); access rows via dataset["train"][i].
  2. Single split       -> load_dataset("stanfordnlp/imdb", split="train")
     Returns just the train split, so no ["train"] indexing is needed.

Each example is {"text": <review>, "label": <int>}, where label 1 = positive,
0 = negative, and -1 = unlabeled (the "unsupervised" split).
"""

from datasets import load_dataset

# Movie reviews dataset
dataset = load_dataset("stanfordnlp/imdb")

print(dataset)                    # shows splits: train / test, and columns
print()
print(dataset["train"][0])        # first training example -> {'text': '...', 'label': 1}
print()
print(dataset["train"].features)  # column types / label meanings
print()
print("Number of training reviews: ", len(dataset["train"]))
print("\n\n\n")


train_dataset = load_dataset("stanfordnlp/imdb", split="train")

# Now ["train"] is not needed, because this dataset only contains training data
print(train_dataset[1000]["text"])
print()

# Label: 1 = positive, 0 = negative
labels = {0: "negative review", 1: "positive review", -1: "unlabeled"}
print("Review status:", labels[train_dataset[1000]["label"]])
