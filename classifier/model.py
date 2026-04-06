"""
BERT-based Fake News Classifier using XLM-RoBERTa (multilingual).
Supports fine-tuning, inference, and confidence scoring.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

LABELS = {0: "REAL", 1: "FAKE", 2: "UNVERIFIED"}
MODEL_NAME = "xlm-roberta-base" 


class FakeNewsDataset(Dataset):
    """
    Expects a list of dicts: [{"text": str, "label": int}, ...]
    label: 0=REAL, 1=FAKE, 2=UNVERIFIED
    Compatible with: LIAR, FakeNewsNet, CLEF2023, MultiFC
    """

    def __init__(self, records: list[dict], tokenizer, max_len: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        text = rec["text"]
        if "body" in rec and rec["body"]:
            text = rec["text"] + " </s> " + rec["body"][:1000] 

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(rec["label"], dtype=torch.long),
        }


class FakeNewsClassifier:
    """
    Fine-tuned XLM-RoBERTa for multilingual fake news detection.

    Usage:
        clf = FakeNewsClassifier()
        clf.fine_tune(train_records, val_records, epochs=3)
        result = clf.predict("Breaking: Scientists discover miracle cure")
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_labels: int = 3,
        device: Optional[str] = None,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)

    def fine_tune(
        self,
        train_records: list[dict],
        val_records: list[dict],
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 2e-5,
        warmup_ratio: float = 0.1,
        max_len: int = 512,
    ) -> dict:
        """Fine-tune on labelled records. Returns training history."""

        train_ds = FakeNewsDataset(train_records, self.tokenizer, max_len)
        val_ds = FakeNewsDataset(val_records, self.tokenizer, max_len)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_dl) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_dl:
                optimizer.zero_grad()
                out = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["label"].to(self.device),
                )
                out.loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += out.loss.item()

            avg_train_loss = total_loss / len(train_dl)

            val_loss, val_acc = self._evaluate(val_dl)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(self.checkpoint_dir / "best")

        return history

    def _evaluate(self, dataloader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in dataloader:
                out = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    labels=batch["label"].to(self.device),
                )
                total_loss += out.loss.item()
                preds = out.logits.argmax(-1)
                correct += (preds == batch["label"].to(self.device)).sum().item()
                total += len(batch["label"])
        return total_loss / len(dataloader), correct / total

    def predict(self, text: str, body: str = "") -> dict:
        """
        Returns:
            {
              "label": "FAKE" | "REAL" | "UNVERIFIED",
              "label_id": int,
              "confidence": float,
              "probabilities": {"REAL": float, "FAKE": float, "UNVERIFIED": float},
            }
        """
        self.model.eval()
        full_text = text + (" </s> " + body[:1000] if body else "")
        enc = self.tokenizer(
            full_text,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        label_id = int(np.argmax(probs))
        return {
            "label": LABELS[label_id],
            "label_id": label_id,
            "confidence": float(probs[label_id]),
            "probabilities": {LABELS[i]: float(p) for i, p in enumerate(probs)},
        }

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        """Efficient batch inference."""
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            encs = self.tokenizer(
                chunk,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**encs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for prob_row in probs:
                label_id = int(np.argmax(prob_row))
                results.append(
                    {
                        "label": LABELS[label_id],
                        "label_id": label_id,
                        "confidence": float(prob_row[label_id]),
                        "probabilities": {
                            LABELS[i]: float(p) for i, p in enumerate(prob_row)
                        },
                    }
                )
        return results


    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "FakeNewsClassifier":
        path = Path(path)
        obj = cls.__new__(cls)
        obj.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        obj.checkpoint_dir = path
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path).to(
            obj.device
        )
        return obj