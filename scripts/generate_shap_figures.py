"""Generate SHAP figures from the trained coupled LightGBM model."""
import argparse
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("LGBM_NUM_THREADS", "4")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import shap
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.engine import SimulationPipeline


def configure_font():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def save_current(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT, "results", "shap_figures"),
        help="Directory where generated SHAP figures are saved.",
    )
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--background-size", type=int, default=150)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_font()
    fig_dir = os.path.abspath(args.output_dir)
    os.makedirs(fig_dir, exist_ok=True)

    pipe = SimulationPipeline()
    pipe.load_and_train()
    art = pipe.compute_shap_analysis(
        sample_size=args.sample_size,
        background_size=args.background_size,
    )
    X = art["X"]
    shap_values = art["shap_values"]

    shap.summary_plot(shap_values, X, max_display=20, show=False, plot_size=(8, 6))
    save_current(os.path.join(fig_dir, "06_shap_summary.png"))

    shap.summary_plot(
        shap_values, X, max_display=20, show=False, plot_type="bar", plot_size=(8, 5)
    )
    save_current(os.path.join(fig_dir, "07_shap_bar.png"))

    dep_feat = pipe.shap_dependence_feature()
    shap.dependence_plot(dep_feat, shap_values, X, show=False, interaction_index=None)
    plt.title(f"SHAP dependence: {dep_feat}")
    save_current(os.path.join(fig_dir, "08_shap_dependence.png"))

    print("Saved SHAP figures:")
    print(os.path.join(fig_dir, "06_shap_summary.png"))
    print(os.path.join(fig_dir, "07_shap_bar.png"))
    print(os.path.join(fig_dir, "08_shap_dependence.png"))
    print("\nTop features:")
    print(art["feature_importance"].head(10).to_string(index=False))
    print("\nFeature-group importance:")
    print(art["group_importance"].to_string(index=False))


if __name__ == "__main__":
    main()
