# 🔍 Fake News Detection System

![Stack](https://img.shields.io/badge/Stack-Python%20%7C%20PyTorch%20%7C%20Transformers%20%7C%20FastAPI-1b2e2b?style=flat-square)
![Model](https://img.shields.io/badge/Model-XLM--RoBERTa%20%7C%20Hugging%20Face-d9c5b2?style=flat-square)
![APIs](https://img.shields.io/badge/APIs-Google%20Fact%20Check%20%7C%20ClaimBuster-4285f4?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-7ecb84?style=flat-square)

An end-to-end misinformation detection system that combines natural language processing, real-time fact-checking APIs, and propagation signal analysis to determine whether a news claim is likely **True**, **False**, or **Unverified**.

Built with **Python**, **PyTorch**, **Transformers**, and **FastAPI**.

** 🌐 Live Demo:** https://fakenews-detector-five.vercel.app/

---

## 1️⃣ Problem Statement

Misinformation spreads rapidly across digital platforms, often reaching large audiences before fact-checking organizations can respond. Traditional detection approaches typically rely on a single signal — such as textual classification — which limits reliability and explainability.

This project builds a **multi-signal misinformation detection system** that combines:

- Transformer-based language understanding
- Real-time fact-checking API verification
- Graph-based propagation signal analysis

By integrating these signals through a fusion layer, the system produces a more robust and explainable credibility verdict for any given news claim.

---

## 2️⃣ Why It Matters

Misinformation has measurable consequences in modern society. False or misleading content influences:

- Elections and political discourse
- Public health decisions
- Financial markets and economic behavior
- Social stability and public trust

Manual fact-checking cannot scale to the volume and velocity of content generated online. Automated systems that assist in flagging suspicious claims support:

- Journalists verifying claims in real time
- Researchers studying information propagation patterns
- Platforms identifying suspicious content before it spreads
- Everyday users assessing the credibility of news sources

This project demonstrates how AI systems can assist in combating misinformation at scale, using a principled multi-signal architecture rather than a black-box classifier.

---

## 3️⃣ Dataset

The classification model is trained on multiple misinformation datasets to improve generalization across domains.

| Dataset | Size | Notes |
|---|---|---|
| LIAR Dataset | ~120K labeled statements | Political claims; six truthfulness categories consolidated into True / False / Unverified |
| FakeNewsNet | Variable | News article-level annotations |
| CLEF Misinformation Datasets | Variable | Cross-lingual claim verification |
| Custom CSV | User-defined | Flexible ingestion for domain-specific claims |

The LIAR dataset serves as the primary benchmark. Its short political statements and noisy labels make it a significantly harder task than standard sentiment or topic classification.

---

## 4️⃣ Methodology

The system uses a **multi-signal pipeline** that analyzes each claim through three independent modules before combining outputs in a fusion layer.

```
Input (title + body + optional graph)
        │
        ├─► [Transformer Classifier]
        │   xlm-roberta-base → FAKE / REAL / UNVERIFIED + confidence score
        │
        ├─► [Fact-Check API]
        │   Google Fact Check Tools API → publisher verdicts
        │   └─ Fallback: ClaimBuster check-worthiness scoring
        │
        └─► [Graph Analyzer]
            Structural + temporal + credibility propagation features
            └─ Optional: PropagationGNN (PyTorch Geometric)
                    │
                    ▼
           [Fusion Layer]
           Weighted ensemble → final verdict + confidence score
```

**Pipeline Steps:**

1. Input claim (title + optional body) is received via the FastAPI endpoint.
2. The transformer classifier encodes the claim and predicts a probability distribution over True / False / Unverified.
3. The fact-checking module queries Google Fact Check Tools API; falls back to ClaimBuster if no match is found.
4. The graph analyzer extracts structural propagation features (credibility signals, temporal dynamics).
5. The fusion layer combines all three signal outputs via weighted ensemble to produce the final verdict.

---

## 5️⃣ Model Architecture

**Core Components**

| Component | Technology |
|---|---|
| Transformer Classifier | XLM-RoBERTa (`xlm-roberta-base`) |
| Fact-Checking | Google Fact Check Tools API + ClaimBuster fallback |
| Propagation Analysis | Graph feature extraction + optional PropagationGNN (PyTorch Geometric) |
| Fusion Layer | Weighted ensemble combining all three signal outputs |
| API Layer | FastAPI (REST) |

**Architecture Diagram**

```
XLM-RoBERTa (fine-tuned)     →  P(True / False / Unverified)
Google FCTAPI / ClaimBuster   →  Publisher verdict signals
Graph Propagation Analyzer    →  Structural credibility features
                                        │
                                        ▼
                              Weighted Fusion Layer
                                        │
                                        ▼
                          Final Verdict + Confidence Score
```

**Project Structure**

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

## 6️⃣ Results

**Classification Performance**

| Metric | Value |
|---|---|
| Validation Accuracy (LIAR) | ~54% |
| Baseline Transformer Range (LIAR) | 55–62% |

Performance is consistent with published baseline transformer results on the LIAR dataset. The multi-signal architecture improves system robustness by supplementing model predictions with external verification signals, reducing false certainty on unseen claims.

The fusion layer demonstrates that combining external fact-checking signals with classifier confidence produces more conservative and explainable verdicts compared to a classifier-only approach.

---

## 7️⃣ Limitations

- **Dataset Size:** The LIAR dataset is relatively small and noisy, which limits generalization to broader domains.
- **Fact-Check API Coverage:** External fact-checking APIs do not cover all claims, particularly recent or niche topics.
- **Simplified Graph Features:** Propagation signal analysis is a prototype implementation; full GNN integration is not deployed in production.
- **Compute Requirements:** Transformer inference requires significant compute resources; latency increases under load.
- **Fundamental Task Complexity:** Misinformation detection is an inherently difficult problem — automated systems can assist but cannot replace human judgment.

---

## 8️⃣ Future Work

- Training on larger and more diverse misinformation datasets
- Expanding multilingual claim detection using multilingual transformer variants
- Integrating Graph Neural Networks (PyTorch Geometric) for deeper propagation modeling
- Adding model explainability features (SHAP, attention visualization)
- Deploying the trained checkpoint on Hugging Face Hub for reproducibility
- Optimizing inference pipeline for real-time detection at scale
- Adding a model evaluation dashboard with precision, recall, and F1-score breakdowns

---

## 9️⃣ How to Run

**1. Clone the Repository**

```bash
git clone https://github.com/yamireddy04/fakenews_detector
cd fakenews_detector
```

**2. Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the API Server**

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

API available at: `http://localhost:8000`

**5. Example Request**

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"title": "Scientists discover coffee cures all diseases", "body": ""}'
```

---

## 🔟 Conclusion

This project presents a multi-signal AI system for fake news detection that moves beyond single-classifier approaches by integrating transformer-based language models, real-time fact-checking APIs, and graph-based propagation signals into a unified fusion pipeline. The architecture prioritizes explainability and robustness over raw accuracy, acknowledging the fundamental complexity of misinformation detection as a task. The result is a scalable and extensible framework that demonstrates principled AI system design — combining multiple evidence sources with principled fusion — as a foundation for more reliable automated verification systems.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Language Model | XLM-RoBERTa (via Hugging Face Transformers) |
| Deep Learning | PyTorch, PyTorch Geometric |
| Fact-Checking | Google Fact Check Tools API, ClaimBuster |
| API | FastAPI |
| Frontend | HTML / CSS / JS (Vercel) |
| Backend Hosting | Render |
