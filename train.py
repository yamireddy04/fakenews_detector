from classifier.model import FakeNewsClassifier
from classifier.datasets import load_liar, split_records

print("Loading data...")
train = load_liar("data/train.tsv")
val = load_liar("data/valid.tsv")

print(f"Train: {len(train)} samples, Val: {len(val)} samples")

clf = FakeNewsClassifier(model_name="xlm-roberta-base", num_labels=3)

print("Starting fine-tuning...")
history = clf.fine_tune(
    train_records=train,
    val_records=val,
    epochs=3,
    batch_size=16,
    lr=2e-5,
)

print("Saving model...")
clf.save("checkpoints/best")
print("Done! Model saved to checkpoints/best")