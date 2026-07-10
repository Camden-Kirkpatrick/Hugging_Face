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
