import os, random
import numpy as np
SEEDS = {
    "nonfire_sampling": 42,
    "split":            2026,
    "kfold":            7,
    "tf":               1234,
    "pso":              99,
}
def set_all(name="tf"):
    if name not in SEEDS:
        raise KeyError(f"Unknown seed '{name}'. Keys: {sorted(SEEDS)}")
    s = SEEDS[name]
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s); np.random.seed(s)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(s)
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            pass
    except ModuleNotFoundError:
        pass
    return s
if __name__ == "__main__":
    print(f"SEEDS = {SEEDS}")
