from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class Consensus(str, Enum):
    SAFE = "SAFE"
    CONFIRMED = "CONFIRMED"
    KNOWN_PATTERN = "KNOWN_PATTERN"
    ZERO_DAY = "ZERO_DAY"


class RuleVerdict(str, Enum):
    BENIGN = "BENIGN"
    SUSPICIOUS = "SUSPICIOUS"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FeedbackVerdict(str, Enum):
    TRUE_POSITIVE = "TRUE_POSITIVE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    FALSE_NEGATIVE = "FALSE_NEGATIVE"


# ── Request schemas ───────────────────────────────────────────────

class FlowRequest(BaseModel):
    features: dict[str, float] = Field(
        ..., description="CIC-IDS2017 feature name → value mapping"
    )


class ThresholdRequest(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)


class FeedbackRequest(BaseModel):
    verdict: FeedbackVerdict
    notes: Optional[str] = None


# ── Response schemas ──────────────────────────────────────────────

class RuleResult(BaseModel):
    verdict: RuleVerdict
    triggered_rules: list[str] = []
    confidence: float = 0.0
    details: dict = {}


class RFResult(BaseModel):
    label: str
    confidence: float
    top_features: list[dict]  # [{feature, shap_value}]


class AEResult(BaseModel):
    score: float
    threshold: float
    label: str  # BENIGN | ANOMALY


class PredictResponse(BaseModel):
    flow_id: str
    rule: RuleResult
    rf: Optional[RFResult] = None      # None when AI pipeline was skipped
    ae: Optional[AEResult] = None
    consensus: Consensus
    severity: Severity
    ai_skipped: bool = False           # True when rule said BENIGN → fast path
    timestamp: datetime


class BatchSummary(BaseModel):
    total: int
    safe: int
    confirmed: int
    known_pattern: int
    zero_day: int
    attack_breakdown: dict[str, int]
    rows: list[dict]


class ExplainResponse(BaseModel):
    flow_id: str
    rf_importance: list[dict]   # global Gini importances (SHAP queued as FE-012)
    ae_recon_error: list[dict]
    natural_language: str


class AlertRow(BaseModel):
    id: int
    timestamp: datetime
    rf_label: str
    rf_conf: float
    ae_score: float
    ae_label: str
    consensus: Consensus
    severity: Severity


class TrainingStage(str, Enum):
    rf  = "rf"
    ae1 = "ae1"
    ae2 = "ae2"
    all = "all"


class TrainingDataSource(str, Enum):
    csv = "csv"
    db  = "db"


class TrainingStartRequest(BaseModel):
    stage: TrainingStage = TrainingStage.all
    data_source: TrainingDataSource = TrainingDataSource.csv


class TrainingRunResponse(BaseModel):
    id: int
    stage: str
    status: str
    data_source: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime


class MetricsResponse(BaseModel):
    precision: Optional[float]           # live (from feedback) or baseline when feedback_count=0
    recall: Optional[float]
    f1: Optional[float]
    false_positive_rate: Optional[float]
    false_alarm_ratio: Optional[float] = None   # FAR = FP/(FP+TP) — false alert rate
    attack_miss_ratio: Optional[float] = None   # AMR = FN/(FN+TP) — missed attack rate
    total_alerts: int
    feedback_count: int
    ae_threshold: float
    baseline_precision: Optional[float] = None  # always from training eval report
    baseline_recall: Optional[float] = None
    baseline_f1: Optional[float] = None
