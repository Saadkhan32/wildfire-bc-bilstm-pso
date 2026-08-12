"""One-command reproduction from the archived package.

Runs, in order, everything that can be rebuilt from the shipped data and
already-trained models (no retraining):

    1. Environment / package checks        (test_reproducibility.py)
    2. Four-model train/test ROC figure    (make_roc_figure.py, trained weights)
    3. Annual wildfire trend figure        (src/fig_wildfire_trend.py)
    4. Seasonal climate composites, Fig.12 (src/specific_humidity_update/Figure12_...)
    5. SHAP beeswarm, Fig. 17              (from data/shap/SHAP_BiLSTM_PSO_values.pkl)
    6. ENSO/PDO teleconnections, Fig. 18   (src/specific_humidity_update/Figure18_...)

Prerequisites: repository (or code.zip) plus data.zip and models.zip unpacked
into the package root — see README.md. Run inside the `wildfire` conda env:

    python reproduce.py

Each step reports PASS/FAIL and the files it wrote; the script continues past
a failed step and exits 0 only if every step passed.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RESULTS = []


def run_step(name, argv, outputs=()):
    print("\n" + "=" * 60)
    print(f"STEP: {name}")
    print("=" * 60)
    rc = subprocess.call([PY] + argv, cwd=HERE)
    made = [o for o in outputs if os.path.exists(os.path.join(HERE, o))]
    ok = (rc == 0) and (len(made) == len(outputs))
    RESULTS.append((name, ok, made))
    return ok


def fig17_beeswarm():
    """SHAP beeswarm (Fig. 17) from the archived SHAP values."""
    name = "SHAP beeswarm (Fig. 17)"
    print("\n" + "=" * 60)
    print(f"STEP: {name}")
    print("=" * 60)
    pkl = os.path.join(HERE, "data", "shap", "SHAP_BiLSTM_PSO_values.pkl")
    out = os.path.join(HERE, "figs", "Fig17_SHAP_beeswarm")
    try:
        import pickle
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        d = pickle.load(open(pkl, "rb"))
        shap_values = np.asarray(d["shap_values"])
        X = np.asarray(d["X_fg"])
        names = ["Specific Humidity" if n == "RH" else n for n in d["feature_names"]]
        shap.summary_plot(
            shap_values, X, feature_names=names, plot_type="dot", show=False,
            max_display=len(names), sort=True, color_bar=True,
            cmap=plt.get_cmap("coolwarm"), plot_size=(10.6, 8.1),
        )
        fig = plt.gcf()
        fig.axes[0].set_xlabel("SHAP value (impact on susceptibility output)", fontsize=11)
        fig.axes[-1].set_ylabel("Feature value (low → high)", fontsize=10)
        os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
        fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
        fig.savefig(out + ".pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
        order = np.argsort(-np.abs(shap_values).mean(0))
        print("Feature order (top -> bottom):", [names[i] for i in order][:5], "...")
        print(f"Wrote {out}.png / .pdf")
        RESULTS.append((name, True, ["figs/Fig17_SHAP_beeswarm.png"]))
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        RESULTS.append((name, False, []))
        return False


FIG18_COLS = ["Tmax", "Precip", "Soil", "PDSI", "AET", "area70",
              "oni_djf", "oni_mam", "oni_jja", "pdo_djf", "pdo_mam", "pdo_jja"]


def stage_fig18_input():
    """The Fig. 18 script reads its annual table from a fixed sub-path; stage
    the packaged data/analysis_dataset.csv there after validating columns."""
    expected = os.path.join(HERE, "reviewer_response_C2_C15", "data", "analysis_dataset.csv")
    if os.path.exists(expected):
        return True
    packaged = os.path.join(HERE, "data", "analysis_dataset.csv")
    if not os.path.exists(packaged):
        print("  Fig. 18 input missing: data/analysis_dataset.csv (unpack data.zip)")
        return False
    import pandas as pd
    cols = list(pd.read_csv(packaged, nrows=1).columns)
    missing = [c for c in FIG18_COLS if c not in cols]
    if missing:
        print(f"  Fig. 18 input lacks required columns: {missing}")
        return False
    os.makedirs(os.path.dirname(expected), exist_ok=True)
    import shutil
    shutil.copyfile(packaged, expected)
    print("  staged packaged data/analysis_dataset.csv for the Fig. 18 script")
    return True


def main():
    run_step("Environment / package checks", ["test_reproducibility.py"])
    run_step("ROC curves from trained models", ["make_roc_figure.py"],
             outputs=["figs/Fig_ROC_4models_train_test.png",
                      "figs/Fig_ROC_4models_train_test.pdf"])
    # run with the package root on sys.path so the script finds figure_style.py
    trend_code = ("import sys, runpy; sys.path.insert(0, '.'); "
                  "runpy.run_path('src/fig_wildfire_trend.py', run_name='__main__')")
    run_step("Annual wildfire trend figure", ["-c", trend_code],
             outputs=["figs/Fig6_wildfire_trend.png", "figs/Fig6_wildfire_trend.pdf"])
    fig12 = "src/specific_humidity_update/Figure12_full_climate_composites_specific_humidity.py"
    run_step("Seasonal climate composites (Fig. 12)",
             ["-c", f"import runpy; runpy.run_path({fig12!r}, run_name='__main__')"],
             outputs=["figs/climate_v2"])
    fig17_beeswarm()
    if stage_fig18_input():
        fig18 = "src/specific_humidity_update/Figure18_ENSO_PDO_teleconnection_specific_humidity.py"
        run_step("ENSO/PDO teleconnections (Fig. 18)",
                 ["-c", f"import runpy; runpy.run_path({fig18!r}, run_name='__main__')"],
                 outputs=["figs/Fig18_ENSO_teleconnection_specifichumidity.png"])
    else:
        RESULTS.append(("ENSO/PDO teleconnections (Fig. 18)", False, []))

    print("\n" + "=" * 60)
    print("REPRODUCTION SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, ok, made in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        for m in made:
            print(f"         -> {m}")
        all_ok &= ok
    print("=" * 60)
    print("All steps passed." if all_ok else "One or more steps failed - see above.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
