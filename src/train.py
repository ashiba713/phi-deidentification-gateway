import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.nn.utils.rnn import pad_sequence


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_train.json"
MODEL_DIR = ROOT / "models" / "distilbert_phi"

MODEL_NAME = "distilbert/distilbert-base-uncased"
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_LENGTH = 192
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def create_label_maps(records):
    labels = sorted({
        tag
        for record in records
        for tag in record["tags"]
    })

    if "O" not in labels:
        labels.insert(0, "O")

    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    return label2id, id2label


class PHIDataset(Dataset):
    def __init__(self, records, tokenizer, label2id):
        self.examples = []

        for record in records:
            words = record["tokens"]
            word_labels = record["tags"]

            encoded = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=MAX_LENGTH
            )

            word_ids = encoded.word_ids()
            aligned_labels = []
            previous_word_id = None

            for word_id in word_ids:
                if word_id is None:
                    aligned_labels.append(-100)
                elif word_id != previous_word_id:
                    aligned_labels.append(
                        label2id[word_labels[word_id]]
                    )
                else:
                    # Ignore continuation subwords in the loss.
                    aligned_labels.append(-100)

                previous_word_id = word_id

            self.examples.append({
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": aligned_labels
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_batch(batch, pad_token_id):
    input_ids = [
        torch.tensor(x["input_ids"], dtype=torch.long)
        for x in batch
    ]

    attention_masks = [
        torch.tensor(x["attention_mask"], dtype=torch.long)
        for x in batch
    ]

    labels = [
        torch.tensor(x["labels"], dtype=torch.long)
        for x in batch
    ]

    return {
        "input_ids": pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        ),
        "attention_mask": pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        ),
        "labels": pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
    }


def main():
    print("Loading synthetic dataset...")
    records = load_data()

    label2id, id2label = create_label_maps(records)

    print("Loading tokenizer and pretrained model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    parameter_count = sum(p.numel() for p in model.parameters())

    print(f"Model parameters: {parameter_count:,}")
    print(f"Labels: {label2id}")

    dataset = PHIDataset(records, tokenizer, label2id)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(SEED)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    collate = lambda batch: collate_batch(
        batch, tokenizer.pad_token_id
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    device = torch.device("cpu")
    model.to(device)

    print(f"Training on: {device}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }

                outputs = model(**batch)
                total_val_loss += outputs.loss.item()

        average_val_loss = total_val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Validation Loss: {average_val_loss:.4f}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    with open(MODEL_DIR / "label_maps.json", "w") as f:
        json.dump({
            "label2id": label2id,
            "id2label": {
                str(k): v for k, v in id2label.items()
            },
            "parameter_count": parameter_count
        }, f, indent=2)

    print("\nTraining completed.")
    print(f"Model saved at: {MODEL_DIR}")
    print(f"Exact parameter count: {parameter_count:,}")


if __name__ == "__main__":
    main()