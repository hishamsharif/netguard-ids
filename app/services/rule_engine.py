"""
Rule-based detection engine — Stage 1 of the 3-stage pipeline.

Applies deterministic heuristics derived from known network attack signatures
(Snort-style rules adapted for CIC-IDS2017 flow features).

Decision:
  BENIGN    → AI pipeline is SKIPPED; verdict = SAFE (fast path)
  SUSPICIOUS → flow is passed to Stage 2 (RF) + Stage 3 (LSTM-AE)

Rules are intentionally conservative (low false-positive bias) so the AI
models handle the ambiguous middle ground.
"""
from __future__ import annotations
from dataclasses import dataclass, field

# Ports associated with high-risk services in CIC-IDS2017 attack scenarios
_HIGH_RISK_PORTS = {
    23,    # Telnet (Brute Force)
    21,    # FTP (Brute Force)
    22,    # SSH (Brute Force / Infiltration)
    445,   # SMB (Infiltration / Botnet)
    135,   # MS-RPC (Botnet)
    137,   # NetBIOS (Botnet)
    139,   # NetBIOS Session (Botnet)
    1433,  # MSSQL (Brute Force)
    3389,  # RDP (Brute Force)
    8080,  # HTTP Alt — only when combined with attack indicators (see R-06)
}

# Features are submitted with their raw CIC-IDS2017 names (may contain spaces)
# Normalise to lowercase-no-space for robust lookup
def _f(features: dict, *names: str) -> float:
    """Look up a feature by any of its name variants, return 0.0 if missing."""
    lookup = {k.strip().lower().replace(" ", "_"): v for k, v in features.items()}
    for n in names:
        key = n.strip().lower().replace(" ", "_")
        if key in lookup:
            return float(lookup[key])
    return 0.0


@dataclass
class RuleResult:
    verdict: str                          # BENIGN | SUSPICIOUS
    triggered_rules: list[str] = field(default_factory=list)
    confidence: float = 0.0              # 0 = confident benign; 1 = confident attack
    details: dict = field(default_factory=dict)


def evaluate(features: dict) -> RuleResult:
    """
    Evaluate a flow against all rules.
    Returns RuleResult(verdict="SUSPICIOUS") as soon as any rule fires.
    Falls through to RuleResult(verdict="BENIGN") if none match.
    """
    triggered: list[str] = []
    details: dict = {}

    # ── Rule R-01: DoS rate threshold ────────────────────────────────
    flow_bytes_s  = _f(features, "Flow Bytes/s", "flow_bytes/s")
    flow_pkts_s   = _f(features, "Flow Packets/s", "flow_packets/s")
    if flow_bytes_s > 1_000_000 or flow_pkts_s > 1_000:
        triggered.append("R-01:high_flow_rate")
        details["flow_bytes_s"] = flow_bytes_s
        details["flow_pkts_s"] = flow_pkts_s

    # ── Rule R-02: SYN flood ─────────────────────────────────────────
    syn_count    = _f(features, "SYN Flag Count", "syn_flag_count")
    flow_dur     = _f(features, "Flow Duration", "flow_duration")
    if syn_count > 50 and flow_dur < 5_000:          # many SYNs in <5ms
        triggered.append("R-02:syn_flood")
        details["syn_count"] = syn_count

    # ── Rule R-03: RST flood ─────────────────────────────────────────
    rst_count = _f(features, "RST Flag Count", "rst_flag_count")
    if rst_count > 10:
        triggered.append("R-03:rst_flood")
        details["rst_count"] = rst_count

    # ── Rule R-04: FIN scan (no SYN) ────────────────────────────────
    fin_count = _f(features, "FIN Flag Count", "fin_flag_count")
    if fin_count > 5 and syn_count == 0:
        triggered.append("R-04:fin_scan")
        details["fin_count"] = fin_count

    # ── Rule R-05: URG flag anomaly ──────────────────────────────────
    fwd_urg = _f(features, "Fwd URG Flags", "fwd_urg_flags")
    bwd_urg = _f(features, "Bwd URG Flags", "bwd_urg_flags")
    if fwd_urg > 0 or bwd_urg > 0:
        triggered.append("R-05:urg_flag_anomaly")
        details["fwd_urg"] = fwd_urg

    # ── Rule R-06: High-risk destination port ────────────────────────
    dst_port = int(_f(features, "Destination Port", "destination_port"))
    psh_count = _f(features, "PSH Flag Count", "psh_flag_count")
    web_attack_port = dst_port in {80, 8080} and (flow_bytes_s > 100_000 or psh_count > 20)
    if dst_port in _HIGH_RISK_PORTS or web_attack_port:
        triggered.append("R-06:high_risk_port")
        details["dst_port"] = dst_port

    # ── Rule R-07: Probe / short-duration near-empty flow ───────────
    total_fwd = _f(features, "Total Fwd Packets", "total_fwd_packets")
    total_bwd = _f(features, "Total Backward Packets", "total_backward_packets")
    if flow_dur > 0 and flow_dur < 200 and total_fwd <= 2 and total_bwd == 0:
        triggered.append("R-07:port_probe")
        details["flow_dur"] = flow_dur

    # ── Rule R-08: Null session (no traffic at all) ──────────────────
    if total_fwd == 0 and total_bwd == 0:
        triggered.append("R-08:null_session")

    # ── Rule R-09: Large payload anomaly ────────────────────────────
    avg_fwd_seg = _f(features, "Avg Fwd Segment Size", "avg_fwd_segment_size")
    if avg_fwd_seg > 64_000:
        triggered.append("R-09:oversized_payload")
        details["avg_fwd_seg"] = avg_fwd_seg

    # ── Rule R-10: PSH+ACK flood ─────────────────────────────────────
    psh_count = _f(features, "PSH Flag Count", "psh_flag_count")
    ack_count = _f(features, "ACK Flag Count", "ack_flag_count")
    if psh_count > 100 and ack_count > 100:
        triggered.append("R-10:psh_ack_flood")
        details["psh_count"] = psh_count

    if not triggered:
        return RuleResult(verdict="BENIGN", confidence=0.0)

    confidence = min(1.0, len(triggered) * 0.25)
    return RuleResult(
        verdict="SUSPICIOUS",
        triggered_rules=triggered,
        confidence=confidence,
        details=details,
    )
