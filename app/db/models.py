from sqlalchemy import Column, Integer, Float, String, DateTime, Text, ForeignKey
from sqlalchemy.sql import func
from app.database import Base


class AlertLog(Base):
    __tablename__ = "alert_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    rf_label = Column(String(64), nullable=False)
    rf_confidence = Column(Float, nullable=False)
    ae_score = Column(Float, nullable=False)
    ae_threshold = Column(Float, nullable=False)
    ae_label = Column(String(16), nullable=False)  # BENIGN | ANOMALY
    consensus = Column(String(16), nullable=False)
    severity = Column(String(16), nullable=False)
    raw_features = Column(Text, nullable=True)      # JSON blob
    rule_verdict = Column(String(16), nullable=True)  # BENIGN | SUSPICIOUS
    rule_triggered = Column(Text, nullable=True)    # JSON list of triggered rule IDs
    ai_skipped = Column(Integer, default=0)         # 1 if AI was bypassed


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    verdict = Column(String(32), nullable=False)  # TRUE_POSITIVE | FALSE_POSITIVE | FALSE_NEGATIVE
    notes = Column(Text, nullable=True)


class TrainingRun(Base):
    """Tracks every ML/DL training job triggered via the API or UI."""
    __tablename__ = "training_run"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stage = Column(String(8), nullable=False)           # rf | ae1 | ae2 | all
    status = Column(String(10), default="queued")       # queued|running|done|failed
    data_source = Column(String(4), default="csv")      # csv | db
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    metrics_json = Column(Text, nullable=True)          # {"rf":{"macro_f1":...}, ...}
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class FeatureLog(Base):
    """
    Scaled + clipped feature vector for every non-rule-bypassed inference.
    alert_id FK ties each row back to AlertLog for the feedback-label bridge.
    Raw material for incremental retraining.
    """
    __tablename__ = "feature_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(
        Integer,
        ForeignKey("alert_log.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    features_json = Column(Text, nullable=False)   # JSON array of floats
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ThresholdLog(Base):
    __tablename__ = "threshold_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    old_threshold = Column(Float, nullable=False)
    new_threshold = Column(Float, nullable=False)
    reason = Column(Text, nullable=True)
