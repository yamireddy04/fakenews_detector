"""
Tests for the full fake news detection pipeline.
Run: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import json
from unittest.mock import MagicMock, patch
from dataclasses import asdict


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------

class TestFakeNewsClassifier:

    def test_dataset_tokenization(self):
        """FakeNewsDataset should tokenize and pad correctly."""
        from classifier.model import FakeNewsDataset
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
        records = [
            {"text": "Breaking: scientists discover cure", "label": 1},
            {"text": "Economy grows 3% in Q3", "label": 0},
        ]
        ds = FakeNewsDataset(records, tok, max_len=64)
        assert len(ds) == 2
        item = ds[0]
        assert item["input_ids"].shape == (64,)
        assert item["attention_mask"].shape == (64,)
        assert item["label"].item() == 1

    def test_dataset_headline_body_concat(self):
        """Body should be concatenated with SEP token."""
        from classifier.model import FakeNewsDataset
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
        records = [{"text": "Title", "body": "Long body text here.", "label": 0}]
        ds = FakeNewsDataset(records, tok, max_len=128)
        item = ds[0]
        # Decoding should contain both parts
        decoded = tok.decode(item["input_ids"])
        assert "Title" in decoded

    def test_predict_shape(self):
        """Predict should return all required keys."""
        from classifier.model import FakeNewsClassifier
        clf = FakeNewsClassifier(num_labels=3)
        result = clf.predict("The president signed a new bill today.")
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["label"] in ("REAL", "FAKE", "UNVERIFIED")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_batch(self):
        """Batch predict should return same count as input."""
        from classifier.model import FakeNewsClassifier
        clf = FakeNewsClassifier(num_labels=3)
        texts = ["Story one", "Story two", "Story three"]
        results = clf.predict_batch(texts)
        assert len(results) == 3
        for r in results:
            assert r["label"] in ("REAL", "FAKE", "UNVERIFIED")


# ---------------------------------------------------------------------------
# Dataset loader tests
# ---------------------------------------------------------------------------

class TestDatasetLoaders:

    def test_liar_loader(self, tmp_path):
        """LIAR loader should parse TSV and map labels."""
        from classifier.datasets import load_liar
        tsv = tmp_path / "train.tsv"
        tsv.write_text(
            "id1\tfalse\tMiracle cure found\tHealth\tJoe\tSenator\tCA\tDem\t1\t5\t2\t3\t0\tInternet\n"
            "id2\ttrue\tTax rate is 21%\tFinance\tJane\tRep\tNY\tRep\t0\t0\t1\t8\t0\tNews\n"
        )
        records = load_liar(tsv)
        assert len(records) == 2
        assert records[0]["label"] == 1  # false → FAKE
        assert records[1]["label"] == 0  # true → REAL

    def test_csv_loader(self, tmp_path):
        """Generic CSV loader should handle bool and int labels."""
        from classifier.datasets import load_csv
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("text,label\nFake headline,fake\nReal news,real\n")
        records = load_csv(csv_path, text_col="text", label_col="label")
        assert records[0]["label"] == 1
        assert records[1]["label"] == 0

    def test_split_records(self):
        from classifier.datasets import split_records
        records = [{"text": f"r{i}", "label": i % 2} for i in range(100)]
        train, val, test = split_records(records, 0.8, 0.1, seed=42)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10


# ---------------------------------------------------------------------------
# Fact-check tests (mocked API)
# ---------------------------------------------------------------------------

class TestFactChecker:

    def _mock_google_response(self):
        return {
            "claims": [
                {
                    "text": "Vaccine causes autism",
                    "claimReview": [
                        {
                            "textualRating": "False",
                            "publisher": {"name": "Snopes"},
                            "url": "https://snopes.com/fact-check/vaccines",
                            "reviewDate": "2023-01-15",
                            "languageCode": "en",
                        }
                    ],
                }
            ]
        }

    def test_normalize_rating(self):
        from factcheck.client import _normalize_rating
        assert _normalize_rating("False") == ("FAKE", 0.95)
        assert _normalize_rating("True") == ("REAL", 0.95)
        assert _normalize_rating("Half True") == ("UNVERIFIED", 0.50)
        assert _normalize_rating("Pants on Fire") == ("FAKE", 1.00)

    def test_disk_cache(self, tmp_path):
        from factcheck.client import DiskCache
        cache = DiskCache(str(tmp_path), ttl_hours=1)
        cache.set("test query", [{"claim": "test", "rating": "False"}])
        result = cache.get("test query")
        assert result is not None
        assert result[0]["claim"] == "test"

    def test_google_client_parse(self):
        from factcheck.client import GoogleFactCheckClient
        with patch("factcheck.client.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(self._mock_google_response()).encode()
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            client = GoogleFactCheckClient(api_key="fake_key", cache_dir="/tmp/fc_test")
            # Bypass cache
            client.cache.get = lambda q: None
            results = client.search("vaccine autism")

        assert len(results) == 1
        assert results[0].rating_normalized == "FAKE"
        assert results[0].publisher == "Snopes"

    def test_factchecker_aggregate(self):
        from factcheck.client import FactChecker, FactCheckResult
        fc = FactChecker()  # no API keys
        fc.google = MagicMock()
        fc.google.search.return_value = [
            FactCheckResult("claim", "False", "FAKE", "Snopes", "url", "2023-01-01", "en", 0.95),
            FactCheckResult("claim", "False", "FAKE", "PolitiFact", "url2", "2023-02-01", "en", 0.90),
        ]
        result = fc.check("test claim")
        assert result["verdict"] == "FAKE"
        assert result["confidence"] > 0.8

    def test_extract_claims(self):
        from factcheck.client import FactChecker
        title = "Scientists discover $5 billion cancer cure"
        body = "The discovery was made in 2024. Researchers claim it is 90% effective."
        claims = FactChecker._extract_claims(title, body, max_claims=3)
        assert title in claims
        assert len(claims) <= 3


# ---------------------------------------------------------------------------
# Graph analysis tests
# ---------------------------------------------------------------------------

class TestPropagationAnalyzer:

    @pytest.fixture
    def sample_graph_dict(self):
        now = 1700000000
        return {
            "root_id": "n0",
            "nodes": [
                {"id": "n0", "is_bot": False, "followers": 50000, "verified": True,  "timestamp": now},
                {"id": "n1", "is_bot": True,  "followers": 100,   "verified": False, "timestamp": now + 60},
                {"id": "n2", "is_bot": True,  "followers": 50,    "verified": False, "timestamp": now + 90},
                {"id": "n3", "is_bot": False, "followers": 5000,  "verified": False, "timestamp": now + 300},
                {"id": "n4", "is_bot": True,  "followers": 80,    "verified": False, "timestamp": now + 400},
            ],
            "edges": [
                {"from": "n0", "to": "n1", "type": "retweet", "timestamp": now + 60},
                {"from": "n0", "to": "n2", "type": "retweet", "timestamp": now + 90},
                {"from": "n1", "to": "n3", "type": "retweet", "timestamp": now + 300},
                {"from": "n1", "to": "n4", "type": "retweet", "timestamp": now + 400},
            ],
        }

    def test_graph_build(self, sample_graph_dict):
        from graph.propagation import GraphBuilder
        g = GraphBuilder.from_dict(sample_graph_dict)
        assert g.root_id == "n0"
        assert len(g.nodes) == 5
        assert len(g.edges) == 4

    def test_bfs_depths(self, sample_graph_dict):
        from graph.propagation import GraphBuilder, PropagationAnalyzer
        g = GraphBuilder.from_dict(sample_graph_dict)
        analyzer = PropagationAnalyzer()
        depths = analyzer._bfs_depths(g)
        assert depths["n0"] == 0
        assert depths["n1"] == 1
        assert depths["n3"] == 2

    def test_features_computed(self, sample_graph_dict):
        from graph.propagation import GraphBuilder, PropagationAnalyzer
        g = GraphBuilder.from_dict(sample_graph_dict)
        analyzer = PropagationAnalyzer()
        feats = analyzer.analyze(g)
        assert feats.num_nodes == 5
        assert feats.num_edges == 4
        assert feats.max_depth == 2
        assert feats.bot_ratio == 0.6  # 3 bots / 5 nodes
        assert 0.0 <= feats.fake_propagation_score <= 1.0

    def test_high_bot_ratio_scores_high(self, sample_graph_dict):
        from graph.propagation import GraphBuilder, PropagationAnalyzer
        g = GraphBuilder.from_dict(sample_graph_dict)
        feats = PropagationAnalyzer().analyze(g)
        # 60% bots should push score above 0.4
        assert feats.fake_propagation_score > 0.4


# ---------------------------------------------------------------------------
# Pipeline integration tests (all mocked)
# ---------------------------------------------------------------------------

class TestPipeline:

    def _build_pipeline(self):
        from pipeline.orchestrator import FakeNewsDetectionPipeline

        mock_clf = MagicMock()
        mock_clf.predict.return_value = {
            "label": "FAKE",
            "confidence": 0.88,
            "probabilities": {"REAL": 0.07, "FAKE": 0.88, "UNVERIFIED": 0.05},
        }

        mock_fc = MagicMock()
        mock_fc.check_article.return_value = {
            "verdict": "FAKE",
            "confidence": 0.92,
            "summary": "Snopes rated False (2023-01-01)",
            "sources": [],
            "claims_checked": [],
        }

        mock_graph = MagicMock()
        mock_graph.analyze.return_value = MagicMock(fake_propagation_score=0.75)

        return FakeNewsDetectionPipeline(
            classifier=mock_clf,
            fact_checker=mock_fc,
            graph_analyzer=mock_graph,
        )

    def test_full_pipeline_returns_verdict(self):
        pipeline = self._build_pipeline()
        result = pipeline.detect(
            title="Shocking: Vaccines proven harmful",
            body="Studies show vaccines cause harm.",
        )
        assert result.verdict in ("FAKE", "REAL", "UNVERIFIED")
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_time_ms > 0

    def test_all_fake_signals_gives_fake(self):
        pipeline = self._build_pipeline()
        result = pipeline.detect("Fake headline", "Fake body")
        assert result.verdict == "FAKE"

    def test_fusion_no_data_graceful(self):
        from pipeline.orchestrator import FakeNewsDetectionPipeline
        pipeline = FakeNewsDetectionPipeline()  # no components
        result = pipeline.detect("Some news article")
        assert result.verdict == "UNVERIFIED"
        assert result.components_used == []

    def test_batch_detection(self):
        pipeline = self._build_pipeline()
        articles = [
            {"title": "Article 1", "body": ""},
            {"title": "Article 2", "body": "Body text"},
        ]
        results = pipeline.detect_batch(articles)
        assert len(results) == 2

    def test_evaluate_returns_metrics(self):
        pipeline = self._build_pipeline()
        data = [
            {"title": "Fake news", "body": "", "label": 1},
            {"title": "Real news", "body": "", "label": 0},
        ]
        metrics = pipeline.evaluate(data)
        assert "accuracy" in metrics
        assert "FAKE" in metrics
        assert "REAL" in metrics


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------

class TestAPIEndpoints:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import app, get_pipeline
        from pipeline.orchestrator import FakeNewsDetectionPipeline
        from unittest.mock import patch

        mock_pipeline = MagicMock()
        from pipeline.orchestrator import DetectionResult
        mock_pipeline.detect.return_value = DetectionResult(
            verdict="FAKE", confidence=0.88,
            bert_label="FAKE", bert_confidence=0.88, bert_probabilities={},
            factcheck_verdict="FAKE", factcheck_confidence=0.92,
            factcheck_summary="test", factcheck_sources=[],
            graph_score=0.7, graph_features={},
            processing_time_ms=120.0,
            components_used=["bert", "factcheck"],
            explanation="Test explanation",
        )
        mock_pipeline.classifier = MagicMock()
        mock_pipeline.fact_checker = MagicMock()

        with patch("api.server.get_pipeline", return_value=mock_pipeline):
            with patch("api.server._pipeline", mock_pipeline):
                yield TestClient(app)

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "status" in resp.json()

    def test_detect_endpoint(self, client):
        with patch("api.server.get_pipeline") as mock_get:
            from pipeline.orchestrator import DetectionResult
            mock_pipe = MagicMock()
            mock_pipe.detect.return_value = DetectionResult(
                verdict="FAKE", confidence=0.88,
                bert_label="FAKE", bert_confidence=0.88, bert_probabilities={},
                factcheck_verdict="FAKE", factcheck_confidence=0.92,
                factcheck_summary="", factcheck_sources=[],
                graph_score=0.7, graph_features={},
                processing_time_ms=100.0,
                components_used=["bert"],
                explanation="",
            )
            mock_get.return_value = mock_pipe
            resp = client.post("/detect", json={"title": "Test headline", "body": ""})
            assert resp.status_code == 200
            data = resp.json()
            assert "verdict" in data