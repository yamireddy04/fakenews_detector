# Fake News Detector

Multilingual fake news detection system combining three complementary signals:

| Component | Model | Signal |
|-----------|-------|--------|
| **BERT Classifier** | `xlm-roberta-base` fine-tuned | Linguistic/semantic patterns |
| **Fact-Check API** | Google Fact Check Tools + ClaimBuster | External verification |
| **Propagation Graph** | NetworkX heuristics + optional GNN (PyG) | Spread pattern analysis |

---

## Architecture

```
Input (title + body + optional graph)
        │
        ├─► [BERT Classifier]        xlm-roberta-base → FAKE/REAL/UNVERIFIED + confidence
        │
        ├─► [Fact-Check API]         Google FCTAPI → claim verdicts from publishers
        │          └─ fallback: ClaimBuster check-worthiness scoring
        │
        └─► [Graph Analyzer]         Structural + temporal + credibility features
                   └─ optional: PropagationGNN (PyTorch Geometric)
                           │
                           ▼
                  [Fusion Layer]     Weighted ensemble → final verdict
```

**Fusion weights (tunable):** BERT 45% · Fact-check 40% · Graph 15%

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt

# Optional: PyTorch Geometric for GNN-based graph classification
pip install torch-geometric
```

### 2. Environment variables

```bash
export GOOGLE_FACTCHECK_API_KEY="your_key"   # https://developers.google.com/fact-check/tools/api
export CLAIMBUSTER_API_KEY="your_key"        # https://idir.uta.edu/claimbuster/ (optional fallback)
export MODEL_CHECKPOINT="checkpoints/best"   # path to fine-tuned model (optional)
```

---

## Training the BERT Classifier

### Option A — Fine-tune on LIAR dataset

```python
from classifier.model import FakeNewsClassifier
from classifier.datasets import load_liar, split_records

records = load_liar("data/liar/train.tsv")
train, val, test = split_records(records)

clf = FakeNewsClassifier(model_name="xlm-roberta-base", num_labels=3)
history = clf.fine_tune(train, val, epochs=3, batch_size=16, lr=2e-5)
clf.save("checkpoints/liar-v1")
```

### Option B — Fine-tune on FakeNewsNet

```python
from classifier.datasets import load_fakenewsnet

records = load_fakenewsnet("data/fakenewsnet/")
train, val, test = split_records(records, train_ratio=0.8)
clf.fine_tune(train, val, epochs=4)
```

### Option C — Your own CSV

```python
from classifier.datasets import load_csv

records = load_csv("data/my_dataset.csv", text_col="title", label_col="fake")
```

### Recommended datasets

| Dataset | Size | Languages | Labels | Download |
|---------|------|-----------|--------|----------|
| LIAR | 12.8K | EN | 6-way | [link](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) |
| FakeNewsNet | 23K+ | EN | Binary | [GitHub](https://github.com/KaiDMML/FakeNewsNet) |
| CLEF 2023 | 20K+ | Multi | Binary | [GitLab](https://gitlab.com/checkthat_lab) |
| MultiFC | 36K | EN | 27-way | [link](https://competitions.codalab.org/competitions/21163) |

---

## Full Pipeline Usage

```python
from classifier.model import FakeNewsClassifier
from factcheck.client import FactChecker
from graph.propagation import PropagationAnalyzer
from pipeline.orchestrator import FakeNewsDetectionPipeline

# Initialize
clf = FakeNewsClassifier.load("checkpoints/best")
fc = FactChecker(google_api_key="YOUR_KEY")
analyzer = PropagationAnalyzer()

pipeline = FakeNewsDetectionPipeline(
    classifier=clf,
    fact_checker=fc,
    graph_analyzer=analyzer,
    weights={"bert": 0.45, "factcheck": 0.40, "graph": 0.15},
)

# Detect
result = pipeline.detect(
    title="Scientists discover miracle weight-loss pill",
    body="A new pill promises to...",
    propagation_graph={
        "root_id": "tweet_001",
        "nodes": [
            {"id": "tweet_001", "is_bot": False, "followers": 5000, "verified": False, "timestamp": 1700000000},
            {"id": "tweet_002", "is_bot": True,  "followers": 12,   "verified": False, "timestamp": 1700000120},
        ],
        "edges": [
            {"from": "tweet_001", "to": "tweet_002", "type": "retweet", "timestamp": 1700000120}
        ],
    }
)

print(result.verdict)       # "FAKE" | "REAL" | "UNVERIFIED"
print(result.confidence)    # 0.87
print(result.explanation)   # Human-readable fusion summary
```

---

## REST API

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Endpoints

```
POST /detect           Full pipeline
POST /detect/batch     Batch (up to 50 articles)
POST /classify         BERT only
POST /factcheck        Fact-check a claim
POST /graph            Graph analysis only
GET  /health           Liveness check
```

### Example

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"title": "Miracle cure discovered", "body": "Details..."}'
```

---

## Graph Analysis

The propagation analyzer extracts these features and computes a `fake_propagation_score`:

| Feature | Fake pattern | Real pattern |
|---------|--------------|--------------|
| `early_spread_ratio` | High (viral burst) | Gradual |
| `bot_ratio` | High | Low |
| `max_breadth / max_depth` | Wide & shallow | Deeper trees |
| `verified_ratio` | Low | Higher |
| `avg_follower_count` | Low (coordinated inauthentic) | Variable |

Based on findings from [Vosoughi et al., Science 2018](https://www.science.org/doi/10.1126/science.aap9559).

### Optional: GNN Classifier (PyTorch Geometric)

```python
from graph.propagation import PropagationGNN, graph_to_pyg, PropagationAnalyzer

gnn = PropagationGNN(in_channels=5, hidden=64, num_classes=2)
pyg_data = graph_to_pyg(graph, PropagationAnalyzer(), label=1)
logits = gnn(pyg_data)
```

---

## Evaluation

```python
# Evaluate on test set
metrics = pipeline.evaluate(test_records)
# {'REAL': {'precision': 0.89, 'recall': 0.91, 'f1': 0.90, 'support': 450},
#  'FAKE': {'precision': 0.87, 'recall': 0.84, 'f1': 0.85, 'support': 380},
#  'UNVERIFIED': {...}, 'accuracy': 0.88}
```

---

## Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
fake-news-detector/
├── classifier/
│   ├── model.py          # XLM-RoBERTa fine-tuning + inference
│   └── datasets.py       # LIAR, FakeNewsNet, CLEF, CSV loaders
├── factcheck/
│   └── client.py         # Google Fact Check + ClaimBuster + caching
├── graph/
│   └── propagation.py    # Graph features + optional GNN (PyG)
├── pipeline/
│   └── orchestrator.py   # Fusion layer + batch + eval
├── api/
│   └── server.py         # FastAPI REST API
├── tests/
│   └── test_pipeline.py  # Full test suite (mocked)
├── requirements.txt
└── README.md
```

---

## Extending

- **Add a new dataset**: implement a loader in `classifier/datasets.py` returning `[{"text", "body", "label"}]`
- **Change fusion weights**: pass `weights={"bert": 0.6, "factcheck": 0.4, "graph": 0.0}` to the pipeline
- **Add a language**: XLM-RoBERTa handles 100 languages natively; pass `language="de"` etc.
- **Swap BERT model**: change `MODEL_NAME` in `classifier/model.py` to e.g. `"facebook/xlm-roberta-xl"`