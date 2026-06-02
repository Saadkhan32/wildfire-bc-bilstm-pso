"""build_reproducibility_audit.py -- Reviewer Comment 8 deliverable.

Writes tables/T_reproducibility_audit.csv: one row per stochastic operation
in the actual training scripts (LSTM/BiLSTM with and without PSO) and the
bootstrap CI script. All training uses SEED = 42.

Run from project root:  python src/build_reproducibility_audit.py
"""
from __future__ import annotations
import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tables" / "T_reproducibility_audit.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
S = 42  # master seed shared by all training scripts and the bootstrap script

COLUMNS = [
    "stochastic_operation",
    "script",
    "library",
    "seed_value",
    "seed_mechanism",
    "deterministic",
    "purpose",
]

ROWS = []

ROWS.append(dict(
    stochastic_operation="RF-RFE feature selection",
    script="LSTM PSO FE.py / BiLSTM PSO FE.py + non-PSO scripts",
    library="scikit-learn RandomForestClassifier + RFE",
    seed_value=S,
    seed_mechanism="explicit random_state=SEED passed to RandomForestClassifier",
    deterministic="Yes",
    purpose="Reproducible feature ranking; runs once non-PSO, 11x in PSO (10 CV folds + final).",
))

ROWS.append(dict(
    stochastic_operation="PSO swarm initialization",
    script="LSTM PSO FE.py / BiLSTM PSO FE.py",
    library="pyswarms.single.GlobalBestPSO",
    seed_value=S,
    seed_mechanism="implicit via np.random.seed(SEED) before pyswarms is called",
    deterministic="Yes (conditional on no other NumPy RNG consumption between np.random.seed and GlobalBestPSO)",
    purpose="Reproducible initial swarm of 6 particles in 10-dim hyperparameter space.",
))

ROWS.append(dict(
    stochastic_operation="PSO particle update steps",
    script="LSTM PSO FE.py / BiLSTM PSO FE.py",
    library="pyswarms.single.GlobalBestPSO",
    seed_value=S,
    seed_mechanism="implicit via np.random.seed(SEED)",
    deterministic="Yes (conditional)",
    purpose="Reproducible PSO trajectory; 6 iterations; c1=1.5, c2=1.5, w=0.7; periodic boundary.",
))

ROWS.append(dict(
    stochastic_operation="70/30 train/test stratified split",
    script="non-PSO LSTM/BiLSTM training script",
    library="sklearn train_test_split",
    seed_value=S,
    seed_mechanism="explicit random_state=RANDOM_STATE",
    deterministic="Yes",
    purpose="Reproducible held-out test partition for early model-comparison runs.",
))

ROWS.append(dict(
    stochastic_operation="85/15 internal train/val split",
    script="all training scripts",
    library="sklearn train_test_split",
    seed_value=S,
    seed_mechanism="explicit random_state=SEED",
    deterministic="Yes",
    purpose="Reproducible validation set for EarlyStopping with restore_best_weights=True.",
))

ROWS.append(dict(
    stochastic_operation="10-fold spatial GroupKFold partition",
    script="LSTM PSO FE.py / BiLSTM PSO FE.py",
    library="sklearn GroupKFold + custom compute_block_ids(lat, lon, grid_km=50)",
    seed_value="N/A",
    seed_mechanism="GroupKFold is deterministic given fixed lat/lon-derived group labels",
    deterministic="Yes",
    purpose="Partition training pool into 10 spatial zones; produces cv_oof_predictions.csv (n=3,351 OOF).",
))

ROWS.append(dict(
    stochastic_operation="Conv1D + LSTM/BiLSTM + Dense weight initialization",
    script="LSTM PSO FE.py / BiLSTM PSO FE.py + non-PSO scripts",
    library="TensorFlow / Keras default glorot_uniform",
    seed_value=S,
    seed_mechanism="tf.keras.utils.set_random_seed(SEED) at top of script",
    deterministic="Algorithmic Yes; bit-level only with enable_op_determinism() (not currently enabled)",
    purpose="Reproducible initialization of all trainable weight tensors.",
))

ROWS.append(dict(
    stochastic_operation="Dropout / recurrent dropout / SpatialDropout1D masks",
    script="all training scripts",
    library="TensorFlow / Keras layers",
    seed_value=S,
    seed_mechanism="tf.keras.utils.set_random_seed(SEED)",
    deterministic="Algorithmic Yes; bit-level only with enable_op_determinism()",
    purpose="Reproducible dropout regularization (PSO best: dropout 0.18-0.36, spatial_drop 0.08-0.25).",
))

ROWS.append(dict(
    stochastic_operation="Mini-batch shuffling per epoch",
    script="all training scripts",
    library="tf.keras Model.fit (shuffle=True)",
    seed_value=S,
    seed_mechanism="tf.keras.utils.set_random_seed(SEED)",
    deterministic="Algorithmic Yes; bit-level only with enable_op_determinism()",
    purpose="Reproducible batch ordering each epoch; PSO best batch in 32/64/128.",
))

ROWS.append(dict(
    stochastic_operation="Non-parametric bootstrap resampling for 95% CIs",
    script="phase_b_bootstrap_ci.py",
    library="NumPy np.random.default_rng",
    seed_value=S,
    seed_mechanism="explicit np.random.default_rng(42) on line 64 (independent RNG instance)",
    deterministic="Yes",
    purpose="1000-resample percentile bootstrap of n=3,351 OOF predictions; produces T_metrics_bootstrap.csv.",
))

with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLUMNS)
    w.writeheader()
    for r in ROWS:
        w.writerow({k: r.get(k, "") for k in COLUMNS})

print("Wrote", OUT)
print("Rows:", len(ROWS), "  Master seed:", S)
