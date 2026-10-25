# NetGuard IDS — Training Pipeline Design

## Architecture: Interface-Driven Pipeline

Every stage in the training pipeline is defined by an **Abstract Base Class (ABC)**.
Concrete implementations can be swapped independently — swap the dataset, the scaler,
the model type, or the evaluation strategy without touching the pipeline orchestrator.

```
training/
├── interfaces/          ← Contracts (ABCs) — stable, never change
│   ├── base_loader.py   → AbstractDataLoader + SplitData
│   ├── base_preprocessor.py → AbstractPreprocessor
│   ├── base_trainer.py  → AbstractTrainer + TrainingHistory
│   └── base_evaluator.py → AbstractEvaluator + EvalResults
│
├── loaders/             ← Data loading implementations
│   ├── cicids2017_loader.py → CICIDS2017Loader (handles all dataset quirks)
│   └── nslkdd_loader.py     → NSLKDDLoader (for AE Phase 1)
│
├── preprocessors/       ← Feature engineering implementations
│   └── cicids_preprocessor.py → VarianceThreshold + RobustScaler pipeline
│
├── trainers/            ← Model training implementations
│   ├── rf_trainer.py    → RFTrainer (Random Forest)
│   └── ae_trainer.py    → NSLKDDAETrainer (Phase 1) + CICIDSAETrainer (Phase 2)
│
├── evaluators/          ← Evaluation implementations
│   └── classification_evaluator.py → ClassificationEvaluator + AnomalyEvaluator
│
└── pipelines/           ← Orchestration
    └── training_pipeline.py → TrainingPipeline (wires stages together)
```

---

## Design Decisions

### Why RobustScaler over StandardScaler?
CIC-IDS2017 contains extreme outliers (DoS flows with millions of packets/s).
`StandardScaler` centres on mean and scales by std-dev — both are dominated by outliers.
`RobustScaler` uses median and IQR, making it insensitive to extreme values while
still normalising the feature space.

### Why class_weight="balanced_subsample" for RF?
`balanced_subsample` rebalances each individual tree's bootstrap sample.
This gives minority attack classes (Infiltration: 36 flows, Heartbleed: 11 flows)
representation in every tree without SMOTE's synthetic sample overhead or risk
of overfitting to interpolated points in sparse regions.

### Why macro-F1 as the headline metric?
CIC-IDS2017 is ~80% benign. Accuracy and weighted-F1 are both inflated by the
dominant class. Macro-F1 averages F1 across all 16 classes equally, giving rare
attack types equal weight. A 0.02 improvement in macro-F1 represents genuine
detection improvement, not just better benign classification.

### Why VarianceThreshold before scaling?
Some CIC-IDS2017 features are constant (e.g. `Fwd Avg Bytes/Bulk` = 0 in all flows).
Passing a zero-variance column to RobustScaler causes a division-by-zero warning
and produces NaN. Filtering near-constants first avoids this and reduces
dimensionality at zero information cost.

### Why 70/15/15 split with stratification?
- 70% train: enough for RF with class_weight and LSTM-AE
- 15% val: used for AE threshold calibration and early stopping signals
- 15% test: held-out, never seen during training or threshold setting
- Stratified: preserves class proportions in all three splits

### Transfer learning rationale
NSL-KDD (41 features, 125k flows) is the canonical IDS benchmark with a
well-balanced benign/attack ratio. Pretraining the LSTM-AE encoder on NSL-KDD
lets it learn general normal-network representations (low-level flow patterns)
before fine-tuning on CIC-IDS2017's larger, higher-dimensional space. Freezing
the encoder for Phase 2 prevents catastrophic forgetting of these representations.

---

## Incremental Versioning Roadmap

### v1.0 — Current (A2 submission target)
Core pipeline with high-impact, essential features:
- [x] Interface layer (ABCs for all 4 stages)
- [x] CICIDS2017Loader with full quirk handling (duplicates, Inf, NaN, labels)
- [x] NSLKDDLoader for AE Phase 1
- [x] CICIDSPreprocessor: VarianceThreshold + RobustScaler
- [x] RFTrainer: balanced_subsample RF, 200 trees, Gini importance export
- [x] NSLKDDAETrainer: LSTM-AE pretrain from scratch
- [x] CICIDSAETrainer: fine-tune with encoder freeze/unfreeze
- [x] ClassificationEvaluator: macro/weighted F1, per-class breakdown, confusion matrix
- [x] AnomalyEvaluator: threshold-based binary eval, recon error distribution
- [x] TrainingPipeline: orchestrates all stages, saves config for reproducibility
- [x] CLI entry points: train_rf.py, pretrain_nslkdd.py, finetune_cicids.py

### v1.1 — Post-A2 (value-added evaluation)
- [ ] ROC/AUC curves per class (sklearn roc_auc_score + matplotlib)
- [ ] SHAP-based post-hoc feature importance in ClassificationEvaluator
- [ ] Learning curves: F1 vs training set size
- [ ] AnomalyEvaluator: add ROC threshold sweep to find optimal θ
- [ ] Export evaluation plots as PNG to reports/ directory
- [ ] EarlyStoppingCallback for AE training (val_loss patience)

### v1.2 — Future (advanced ML engineering)
- [ ] StratifiedKFold cross-validation wrapper
- [ ] RandomizedSearchCV hyperparameter tuning for RF
- [ ] SMOTE oversampling option (imbalanced-learn) in RFTrainer
- [ ] SelectFromModel feature selector (RF importance → prune to top-K)
- [ ] UNSW-NB15Loader (new dataset — drop-in, same interface)
- [ ] ModelRegistry: track all trained models + metrics in a SQLite table
- [ ] MLflow experiment tracking integration

---

## Running the Pipeline

```bash
# Activate environment
conda activate dl_env
cd dev/netguard-ids

# Step 1: Train Random Forest
python -m training.train_rf --data data/cicids2017/ --trees 200

# Step 2: Pretrain LSTM-AE on NSL-KDD
python -m training.pretrain_nslkdd --data data/nslkdd/KDDTrain+.txt --epochs 30

# Step 3: Fine-tune LSTM-AE on CIC-IDS2017
python -m training.finetune_cicids --data data/cicids2017/ --epochs 20

# After Step 3: copy AE_THRESHOLD from output into .env
```

## Adding a New Dataset

Implement `AbstractDataLoader`:
```python
class UNSWNB15Loader(AbstractDataLoader):
    def load(self, path): ...
    def get_label_column(self): return "label"
    def get_feature_columns(self, df): ...
    def split(self, df, ...): ...
```
Pass it to `TrainingPipeline` — nothing else changes.

## Adding a New Preprocessor

Implement `AbstractPreprocessor`:
```python
class PCAPreprocessor(AbstractPreprocessor):
    def fit(self, X, y=None): ...
    def transform(self, X): ...
    def get_feature_names_out(self): ...
    def save(self, path): ...
    def load(cls, path): ...
```
