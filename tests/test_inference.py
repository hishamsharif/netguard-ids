import numpy as np
import torch
import pytest
from app.models.lstm_ae import LSTMAutoencoder
from app.services.consensus import compute_consensus
from app.services import rule_engine
from app.models.schemas import RFResult, AEResult


def make_model(n_features=10):
    return LSTMAutoencoder(n_features=n_features, hidden_dim=16, num_layers=1)


def test_lstm_ae_forward_shape():
    model = make_model(10)
    x = torch.randn(4, 3, 10)  # batch=4, window=3, features=10
    out = model(x)
    assert out.shape == x.shape


def test_lstm_ae_reconstruction_error_shape():
    model = make_model(10)
    x = torch.randn(8, 1, 10)
    errors = model.reconstruction_error(x)
    assert errors.shape == (8,)


def test_lstm_ae_freeze_unfreeze():
    model = make_model(10)
    model.freeze_encoder()
    for p in model.encoder.parameters():
        assert not p.requires_grad
    model.unfreeze_encoder()
    for p in model.encoder.parameters():
        assert p.requires_grad


def test_consensus_safe():
    rf = RFResult(label="BENIGN", confidence=0.99, top_features=[])
    ae = AEResult(score=0.005, threshold=0.023, label="BENIGN")
    consensus, severity = compute_consensus(rf, ae)
    assert consensus.value == "SAFE"
    assert severity.value == "LOW"


def test_consensus_confirmed():
    rf = RFResult(label="DoS Hulk", confidence=0.95, top_features=[])
    ae = AEResult(score=0.05, threshold=0.023, label="ANOMALY")
    consensus, severity = compute_consensus(rf, ae)
    assert consensus.value == "CONFIRMED"
    assert severity.value == "CRITICAL"


def test_consensus_zero_day():
    rf = RFResult(label="BENIGN", confidence=0.92, top_features=[])
    ae = AEResult(score=0.08, threshold=0.023, label="ANOMALY")
    consensus, severity = compute_consensus(rf, ae)
    assert consensus.value == "ZERO_DAY"
    assert severity.value == "HIGH"


# ── Rule Engine tests ─────────────────────────────────────────────

def test_rule_engine_benign_normal_flow():
    features = {
        "Flow Duration": 50000,
        "Total Fwd Packets": 10,
        "Total Backward Packets": 8,
        "Flow Bytes/s": 1000,
        "Flow Packets/s": 5,
        "SYN Flag Count": 1,
        "FIN Flag Count": 1,
        "RST Flag Count": 0,
        "PSH Flag Count": 2,
        "ACK Flag Count": 5,
        "Fwd URG Flags": 0,
        "Bwd URG Flags": 0,
        "Destination Port": 443,
        "Avg Fwd Segment Size": 512,
    }
    result = rule_engine.evaluate(features)
    assert result.verdict == "BENIGN"
    assert result.triggered_rules == []


def test_rule_engine_dos_rate():
    result = rule_engine.evaluate({"Flow Bytes/s": 2_000_000, "Flow Packets/s": 5})
    assert result.verdict == "SUSPICIOUS"
    assert any("R-01" in r for r in result.triggered_rules)


def test_rule_engine_syn_flood():
    result = rule_engine.evaluate({
        "SYN Flag Count": 100, "Flow Duration": 1000,
        "FIN Flag Count": 0, "Total Fwd Packets": 100
    })
    assert result.verdict == "SUSPICIOUS"
    assert any("R-02" in r for r in result.triggered_rules)


def test_rule_engine_high_risk_port():
    result = rule_engine.evaluate({"Destination Port": 23})
    assert result.verdict == "SUSPICIOUS"
    assert any("R-06" in r for r in result.triggered_rules)


def test_rule_engine_urg_flag():
    result = rule_engine.evaluate({"Fwd URG Flags": 1})
    assert result.verdict == "SUSPICIOUS"
    assert any("R-05" in r for r in result.triggered_rules)


def test_rule_engine_confidence_scales_with_rule_count():
    # Multiple rules fired → higher confidence
    result = rule_engine.evaluate({
        "Flow Bytes/s": 2_000_000,
        "SYN Flag Count": 100,
        "Flow Duration": 1000,
        "Fwd URG Flags": 1,
        "Destination Port": 23,
    })
    assert result.verdict == "SUSPICIOUS"
    assert result.confidence >= 0.5  # at least 2 rules × 0.25


def test_rule_engine_null_session():
    result = rule_engine.evaluate({"Total Fwd Packets": 0, "Total Backward Packets": 0})
    assert result.verdict == "SUSPICIOUS"
    assert any("R-08" in r for r in result.triggered_rules)
