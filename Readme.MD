# Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)

An end-to-end misinformation detection system that combines **natural language processing, real-time fact-checking APIs, and propagation signal analysis** to determine whether a news claim is likely **True, False, or Unverified**.

The system integrates a **fine-tuned multilingual transformer model**, external fact-checking signals, and graph-based propagation features into a **fusion layer** that produces a confidence-weighted verdict.

---

### Live Demo

Frontend:  
https://fakenews-detector-five.vercel.app/

API Endpoint:  
https://fakenews-detector-yi5b.onrender.com

Repository:  
https://github.com/yamireddy04/fakenews_detector

---

## Problem Statement

Misinformation spreads rapidly across digital platforms, often reaching large audiences before fact-checking organizations can respond. Traditional approaches to detecting fake news typically rely on a **single signal**, such as textual classification, which limits reliability.

This project aims to address the problem by building a **multi-signal misinformation detection system** that combines:

- Transformer-based language understanding
- Real-time fact-checking APIs
- Graph-based propagation analysis

By integrating these signals, the system can produce a **more robust and explainable verdict** about the credibility of a news claim.

---

## Why It Matters

Misinformation has serious consequences in modern societies. False or misleading information can influence:

- Elections and political discourse
- Public health decisions
- Financial markets
- Social stability

Manual fact-checking cannot scale to the volume and speed of information generated online. Automated tools that assist in identifying potentially misleading content can support:

- Journalists verifying claims
- Researchers studying information flows
- Platforms flagging suspicious content
- Users evaluating the credibility of news

This project demonstrates how **AI systems can assist in combating misinformation at scale**.

---

## Dataset

The classification model is trained using multiple misinformation datasets to improve generalization.

Primary datasets include:

***LIAR Dataset***

- ~12,000 labeled political statements
- Originally contains six truthfulness categories
- Labels consolidated into three categories for this system:
  - True
  - False
  - Unverified

***Additional Sources***

- FakeNewsNet
- CLEF misinformation datasets
- Custom CSV ingestion support

These datasets contain **short claims, political statements, and news headlines**, which makes the task significantly harder than typical text classification tasks.

---

## Methodology

The system uses a **multi-signal pipeline** that analyzes a claim through three independent modules.

### Transformer-Based Text Classification

A multilingual transformer model predicts the probability that a statement is fake or real based on linguistic features and contextual understanding.

### Fact-Checking Verification

The pipeline queries external fact-checking services to determine whether the claim has already been verified by trusted publishers.

### Propagation Signal Analysis

Graph-based features analyze structural patterns that often characterize the spread of misinformation, including credibility signals and temporal propagation dynamics.

The outputs from these components are combined in a **fusion layer** to generate the final verdict.

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

Key components:

- ***XLM-RoBERTa transformer classifier***
- ***Google Fact Check Tools API integration***
- ***ClaimBuster fallback scoring***
- ***Graph-based propagation analysis***
- ***Weighted ensemble fusion***

---

## Results

Evaluation on the LIAR dataset shows that misinformation detection remains a **challenging NLP task**.

Observed performance:

Validation accuracy: approximately **54%**

This performance is consistent with baseline transformer results on the dataset, which typically range between **55% and 62%**.

The multi-signal architecture improves system robustness by combining model predictions with external verification signals.

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

````

---

## How to Run

### Clone the Repository

```bash
git clone https://github.com/yamireddy04/fakenews_detector
cd fakenews_detector
````

---

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Run the API Server

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:

```
http://localhost:8000
```

---

### Example Request

```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"title": "Scientists discover coffee cures all diseases", "body": ""}'
```

---

## Limitations

Despite promising results, the system has limitations:

* The LIAR dataset is relatively small and noisy.
* Fact-checking APIs do not cover all claims.
* Graph propagation signals are simplified in the prototype.
* Transformer inference requires significant compute resources.

Misinformation detection is fundamentally complex and cannot be solved perfectly by automated systems.

---

## Future Work

Potential improvements include:

* Training on larger misinformation datasets
* Expanding multilingual claim detection
* Integrating Graph Neural Networks for deeper propagation modeling
* Adding model explainability features
* Deploying the trained checkpoint on Hugging Face Hub
* Optimizing inference for faster real-time detection

---

## Summary

This project presents a **multi-signal AI system for fake news detection** that integrates:

* Transformer-based language models
* Real-time fact-checking APIs
* Graph-based propagation signals

Instead of relying on a single classifier, the system combines multiple verification signals to produce **confidence-weighted credibility assessments**.

The result is a scalable and extensible framework for **automated misinformation detection and verification**.

---
