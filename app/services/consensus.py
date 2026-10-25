from app.models.schemas import Consensus, Severity, RFResult, AEResult

_BENIGN_LABELS = {"BENIGN", "benign", "Benign"}


def compute_consensus(rf: RFResult, ae: AEResult) -> tuple[Consensus, Severity]:
    """
    Dual-model consensus engine.

    RF label  | AE label | Consensus
    ----------|----------|------------------
    BENIGN    | BENIGN   | SAFE
    BENIGN    | ANOMALY  | ZERO_DAY
    ATTACK    | ANOMALY  | CONFIRMED
    ATTACK    | BENIGN   | KNOWN_PATTERN
    """
    rf_benign = rf.label in _BENIGN_LABELS
    ae_benign = ae.label == "BENIGN"

    if rf_benign and ae_benign:
        return Consensus.SAFE, Severity.LOW
    if rf_benign and not ae_benign:
        return Consensus.ZERO_DAY, Severity.HIGH
    if not rf_benign and not ae_benign:
        return Consensus.CONFIRMED, Severity.CRITICAL
    # RF=ATTACK, AE=BENIGN — known signature, low reconstruction deviation
    return Consensus.KNOWN_PATTERN, _severity_from_confidence(rf.confidence)


def _severity_from_confidence(conf: float) -> Severity:
    if conf >= 0.9:
        return Severity.HIGH
    if conf >= 0.7:
        return Severity.MEDIUM
    return Severity.LOW
