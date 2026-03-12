"""
Pipeline Orchestrator — fuses BERT classifier, Fact-Check API, and Graph Analysis.

Fusion strategy: weighted ensemble with adaptive confidence scaling.
If any component lacks data, it gracefully degrades and reweights others.
"""

import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    # Final verdict
    verdict: str               # "FAKE" | "REAL" | "UNVERIFIED"
    confidence: float          # 0.0–1.0

    # Per-component outputs
    bert_label: str
    bert_confidence: float
    bert_probabilities: dict

    factcheck_verdict: str
    factcheck_confidence: float
    factcheck_summary: str
    factcheck_sources: list

    graph_score: float         # 0=real, 1=fake (or -1 if no graph data)
    graph_features: dict

    # Meta
    processing_time_ms: float
    components_used: list[str]
    explanation: str


# ---------------------------------------------------------------------------
# Fusion weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "bert": 0.45,
    "factcheck": 0.40,
    "graph": 0.15,
}

# Map string labels to numeric fake probability
LABEL_TO_FAKE_PROB = {
    "FAKE": 1.0,
    "UNVERIFIED": 0.5,
    "REAL": 0.0,
    "NO_DATA": None,  # excluded from fusion
}

# ---------------------------------------------------------------------------
# Verdict thresholds
# These control how confident the model needs to be before saying FAKE or REAL
# Raising FAKE_THRESHOLD reduces false positives (real news called fake)
# Lowering REAL_THRESHOLD reduces false negatives (fake news called real)
# ---------------------------------------------------------------------------

FAKE_THRESHOLD = 0.72   # must be this confident to call FAKE (was 0.65)
REAL_THRESHOLD = 0.30   # must be this confident to call REAL (was 0.35)

# Minimum BERT confidence required to trust its verdict
# Below this, we soften the prediction toward UNVERIFIED
MIN_BERT_CONFIDENCE = 0.60


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class FakeNewsDetectionPipeline:
    """
    Full pipeline: text → BERT → Fact-Check → Graph → fused verdict.

    Usage:
        pipeline = FakeNewsDetectionPipeline(
            classifier=FakeNewsClassifier.load("checkpoints/best"),
            fact_checker=FactChecker(google_api_key="..."),
            graph_analyzer=PropagationAnalyzer(),
        )
        result = pipeline.detect(
            title="...",
            body="...",
            propagation_graph=graph_dict,   # optional
        )
    """

    def __init__(
        self,
        classifier=None,
        fact_checker=None,
        graph_analyzer=None,
        weights: dict = None,
    ):
        self.classifier = classifier
        self.fact_checker = fact_checker
        self.graph_analyzer = graph_analyzer
        self.weights = weights or DEFAULT_WEIGHTS

        available = []
        if classifier: available.append("bert")
        if fact_checker: available.append("factcheck")
        if graph_analyzer: available.append("graph")
        logger.info(f"Pipeline initialized with components: {available}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def detect(
        self,
        title: str,
        body: str = "",
        propagation_graph: Optional[dict] = None,
        language: str = "en",
    ) -> DetectionResult:
        t0 = time.perf_counter()

        # --- 1. BERT classifier ---
        bert_result = self._run_bert(title, body)

        # --- 2. Fact-check ---
        fc_result = self._run_factcheck(title, body, language)

        # --- 3. Graph analysis ---
        graph_result, graph_feats = self._run_graph(propagation_graph)

        # --- 4. Fusion ---
        verdict, confidence, components_used = self._fuse(
            bert_result, fc_result, graph_result
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        explanation = self._explain(
            verdict, bert_result, fc_result, graph_result, components_used
        )

        return DetectionResult(
            verdict=verdict,
            confidence=round(confidence, 3),
            bert_label=bert_result.get("label", "UNKNOWN"),
            bert_confidence=bert_result.get("confidence", 0.0),
            bert_probabilities=bert_result.get("probabilities", {}),
            factcheck_verdict=fc_result.get("verdict", "NO_DATA"),
            factcheck_confidence=fc_result.get("confidence", 0.0),
            factcheck_summary=fc_result.get("summary", ""),
            factcheck_sources=fc_result.get("sources", []),
            graph_score=graph_result,
            graph_features=asdict(graph_feats) if graph_feats else {},
            processing_time_ms=round(elapsed_ms, 1),
            components_used=components_used,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Component runners (each returns a normalized dict or fallback)
    # ------------------------------------------------------------------

    def _run_bert(self, title: str, body: str) -> dict:
        if not self.classifier:
            return {"label": "UNKNOWN", "confidence": 0.0, "probabilities": {}}
        try:
            return self.classifier.predict(title, body)
        except Exception as e:
            logger.error(f"BERT inference failed: {e}")
            return {"label": "UNKNOWN", "confidence": 0.0, "probabilities": {}}

    def _run_factcheck(self, title: str, body: str, language: str) -> dict:
        if not self.fact_checker:
            return {"verdict": "NO_DATA", "confidence": 0.0, "summary": "", "sources": []}
        try:
            return self.fact_checker.check_article(title, body, language)
        except Exception as e:
            logger.error(f"Fact-check failed: {e}")
            return {"verdict": "NO_DATA", "confidence": 0.0, "summary": "", "sources": []}

    def _run_graph(self, graph_dict: Optional[dict]):
        from graph.propagation import GraphBuilder, PropagationAnalyzer, GraphFeatures
        if not self.graph_analyzer or not graph_dict:
            return -1.0, None
        try:
            g = GraphBuilder.from_dict(graph_dict)
            feats = self.graph_analyzer.analyze(g)
            return feats.fake_propagation_score, feats
        except Exception as e:
            logger.error(f"Graph analysis failed: {e}")
            return -1.0, None

    # ------------------------------------------------------------------
    # Fusion logic
    # ------------------------------------------------------------------

    def _fuse(
        self,
        bert: dict,
        fc: dict,
        graph_score: float,
    ) -> tuple[str, float, list[str]]:
        """
        Weighted fusion of fake probabilities from each component.
        Components with NO_DATA are excluded and weights redistributed.

        Key fix: BERT confidence below MIN_BERT_CONFIDENCE gets dampened
        toward 0.5 (uncertain) so low-confidence BERT predictions don't
        dominate the verdict and cause false positives.
        """
        components = {}

        # BERT contribution — dampen low confidence predictions
        bert_label = bert.get("label", "UNKNOWN")
        bert_conf = bert.get("confidence", 0.0)
        if bert_label in LABEL_TO_FAKE_PROB and LABEL_TO_FAKE_PROB[bert_label] is not None:
            p = LABEL_TO_FAKE_PROB[bert_label]

            # If BERT confidence is low, pull prediction toward 0.5 (uncertain)
            # This prevents a 55% confident FAKE prediction from dominating
            if bert_conf < MIN_BERT_CONFIDENCE:
                # Dampen: interpolate between 0.5 and the actual prediction
                dampen_factor = bert_conf / MIN_BERT_CONFIDENCE
                p_scaled = 0.5 + (p - 0.5) * bert_conf * dampen_factor
            else:
                p_scaled = 0.5 + (p - 0.5) * bert_conf

            components["bert"] = p_scaled

        # Fact-check contribution
        fc_verdict = fc.get("verdict", "NO_DATA")
        fc_conf = fc.get("confidence", 0.0)
        if fc_verdict in LABEL_TO_FAKE_PROB and LABEL_TO_FAKE_PROB[fc_verdict] is not None:
            p = LABEL_TO_FAKE_PROB[fc_verdict]
            p_scaled = 0.5 + (p - 0.5) * fc_conf
            components["factcheck"] = p_scaled

        # Graph contribution
        if graph_score >= 0:
            components["graph"] = graph_score

        if not components:
            return "UNVERIFIED", 0.0, []

        # Redistribute weights to active components only
        active_weight_sum = sum(self.weights.get(k, 0.1) for k in components)
        weighted_fake_prob = sum(
            v * self.weights.get(k, 0.1) / active_weight_sum
            for k, v in components.items()
        )

        # Apply verdict thresholds
        # Higher FAKE_THRESHOLD = less likely to wrongly call real news fake
        if weighted_fake_prob >= FAKE_THRESHOLD:
            verdict = "FAKE"
            confidence = weighted_fake_prob
        elif weighted_fake_prob <= REAL_THRESHOLD:
            verdict = "REAL"
            confidence = 1.0 - weighted_fake_prob
        else:
            verdict = "UNVERIFIED"
            # Confidence reflects how close to center (0.5) the prediction is
            confidence = 1.0 - abs(weighted_fake_prob - 0.5) * 2

        return verdict, confidence, list(components.keys())

    def _explain(
        self,
        verdict: str,
        bert: dict,
        fc: dict,
        graph_score: float,
        components: list[str],
    ) -> str:
        parts = [f"Verdict: {verdict}."]
        if "bert" in components:
            bert_conf = bert.get("confidence", 0)
            low_conf_note = " (low confidence — treated as uncertain)" if bert_conf < MIN_BERT_CONFIDENCE else ""
            parts.append(
                f"BERT ({bert.get('label')}, conf={bert_conf:.0%}{low_conf_note})"
            )
        if "factcheck" in components:
            parts.append(
                f"Fact-check ({fc.get('verdict')}, conf={fc.get('confidence', 0):.0%}): "
                f"{fc.get('summary', '')[:120]}"
            )
        if "graph" in components and graph_score >= 0:
            parts.append(
                f"Propagation graph fake-score={graph_score:.2f} "
                f"({'suspicious' if graph_score > 0.6 else 'normal'} spread pattern)"
            )
        if not components:
            parts.append("No data available from any component.")
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Batch detection
    # ------------------------------------------------------------------

    def detect_batch(self, articles: list[dict]) -> list[DetectionResult]:
        """
        articles: [{"title": str, "body": str, "graph": dict|None}]
        """
        results = []
        for art in articles:
            r = self.detect(
                title=art.get("title", ""),
                body=art.get("body", ""),
                propagation_graph=art.get("graph"),
                language=art.get("language", "en"),
            )
            results.append(r)
        return results

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, labelled_articles: list[dict]) -> dict:
        """
        labelled_articles: [{"title", "body", "label": 0|1, ...}]
        Returns: precision, recall, F1, accuracy per class
        """
        y_true, y_pred = [], []
        label_map = {0: "REAL", 1: "FAKE", 2: "UNVERIFIED"}

        for art in labelled_articles:
            res = self.detect(art.get("title", ""), art.get("body", ""))
            y_true.append(label_map.get(art["label"], "UNKNOWN"))
            y_pred.append(res.verdict)

        labels = ["REAL", "FAKE", "UNVERIFIED"]
        metrics = {}
        for lbl in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            metrics[lbl] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}

        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
        metrics["accuracy"] = accuracy
        return metrics