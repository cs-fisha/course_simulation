"""
多环境应力耦合失效机理可视化仿真系统
课程：系统建模与仿真技术
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import weibull_min

_cjk_paths = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]
for p in _cjk_paths:
    if os.path.exists(p):
        fm.fontManager.addfont(p)
        plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

from engine import (
    SimulationPipeline, DegradationHMM, STRESS_CN, RUL_CLIP, evaluate
)

@st.cache_resource
def get_pipeline():
    pipe = SimulationPipeline()
    pipe.load_and_train()
    return pipe


def main():
    st.set_page_config(page_title="多环境应力失效机理仿真", layout="wide")
    st.title("多环境应力耦合失效机理可视化仿真系统")
    st.caption("基于 NASA C-MAPSS FD004 数据集 | LightGBM + Weibull + HMM + SHAP")

    with st.spinner("正在加载数据并训练模型（首次约15秒）..."):
        pipe = get_pipeline()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Weibull可靠性仿真", "退化状态演化", "RUL预测对比",
        "蒙特卡洛寿命仿真", "模型评估指标"
    ])

    with tab1:
        render_weibull_tab(pipe)
    with tab2:
        render_hmm_tab(pipe)
    with tab3:
        render_rul_tab(pipe)
    with tab4:
        render_monte_carlo_tab(pipe)
    with tab5:
        render_metrics_tab(pipe)


def render_weibull_tab(pipe):
    st.subheader("Weibull可靠度曲线仿真")
    st.markdown("调节应力水平，观察不同环境应力对系统可靠度的影响。")

    col1, col2 = st.columns([1, 3])
    with col1:
        stress_key = st.selectbox("选择应力类型", list(STRESS_CN.keys()),
                                  format_func=lambda x: STRESS_CN[x])
        show_levels = st.checkbox("显示高/中/低应力分组", value=True)
        t_max = st.slider("时间范围（周期）", 100, 800, 500)

    with col2:
        t_range = np.linspace(1, t_max, 500)
        data = pipe.weibull_reliability_curve(stress_key, t_range)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data["t"], data["overall"], "b-", lw=2.5, label="总体可靠度 R(t)")
        if show_levels:
            colors = {"低": "#27ae60", "中": "#f39c12", "高": "#c0392b"}
            for lbl in ["低", "中", "高"]:
                if lbl in data:
                    ax.plot(data["t"], data[lbl], "--", color=colors[lbl],
                            lw=1.8, label=f"{lbl}应力水平")
        ax.axhline(0.8, color="red", ls=":", alpha=0.5, label="可靠度阈值 0.8")
        ax.set_xlabel("运行周期 t")
        ax.set_ylabel("可靠度 R(t)")
        ax.set_title(f"{STRESS_CN[stress_key]} — Weibull可靠度曲线")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        st.pyplot(fig)
        plt.close()

    mdl = pipe.single_models[stress_key]
    sh, _, sc = mdl.weibull_params
    c1, c2, c3 = st.columns(3)
    c1.metric("形状参数 β", f"{sh:.2f}")
    c2.metric("尺度参数 η", f"{sc:.1f}")
    c3.metric("平均寿命", f"{mdl.mean_life():.0f} 周期")


def render_hmm_tab(pipe):
    st.subheader("HMM退化状态演化仿真")
    st.markdown("选择发动机单元，观察其退化状态随时间的演化轨迹。")

    col1, col2 = st.columns([1, 3])
    with col1:
        unit_id = st.selectbox("选择发动机编号", pipe.unit_ids[:50])
        show_hi = st.checkbox("叠加健康指数曲线", value=True)
        predict_steps = st.slider("前向预测步数", 50, 200, 100)

    traj = pipe.get_unit_trajectory(unit_id)
    if traj is None:
        st.warning("未找到该发动机数据")
        return

    with col2:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

        ax = axes[0]
        ax.plot(traj["cycles"], traj["hmm_states"], "m-", lw=1.5, label="HMM状态路径")
        if show_hi:
            ax2 = ax.twinx()
            ax2.plot(traj["cycles"], traj["hi"], "g--", alpha=0.6, lw=1, label="健康指数HI")
            ax2.set_ylabel("健康指数", color="green")
            ax2.legend(loc="upper left")
        ax.set_yticks(range(5))
        ax.set_yticklabels(DegradationHMM.LABELS, fontsize=9)
        ax.set_xlabel("运行周期")
        ax.set_title(f"发动机 #{unit_id} 退化状态演化（Viterbi解码）")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        obs_half = traj["obs"][:len(traj["obs"])//2]
        preds = pipe.hmm.forward_predict(obs_half, steps=predict_steps)
        ax = axes[1]
        for s in range(5):
            ax.plot(range(predict_steps), preds[:, s], lw=1.5,
                    label=DegradationHMM.LABELS[s])
        ax.axhline(0.5, color="red", ls=":", alpha=0.5)
        ax.set_xlabel("未来周期")
        ax.set_ylabel("状态概率")
        ax.set_title(f"前向预测（观测前{len(obs_half)}步后预测{predict_steps}步）")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("状态转移概率矩阵")
    import pandas as pd
    A_df = pd.DataFrame(pipe.hmm.A, index=DegradationHMM.LABELS,
                        columns=DegradationHMM.LABELS)
    st.dataframe(A_df.style.format("{:.4f}").background_gradient(cmap="YlOrRd"),
                 width="stretch")


def render_rul_tab(pipe):
    st.subheader("RUL预测对比")
    st.markdown("对比单应力模型与多应力耦合模型的剩余寿命预测效果。")

    true_rul, single_preds, coupled_pred = pipe.get_test_predictions()

    col1, col2 = st.columns([1, 3])
    with col1:
        show_models = st.multiselect(
            "显示模型",
            ["温度应力", "机械应力", "热负荷应力", "多应力耦合"],
            default=["多应力耦合"]
        )
        plot_type = st.radio("图表类型", ["时序对比", "散点图"])

    model_map = {"温度应力": "altitude", "机械应力": "mach",
                 "热负荷应力": "tra", "多应力耦合": "coupled"}
    colors = {"温度应力": "#c0392b", "机械应力": "#27ae60",
              "热负荷应力": "#2980b9", "多应力耦合": "#8e44ad"}

    with col2:
        fig, ax = plt.subplots(figsize=(12, 6))
        if plot_type == "时序对比":
            ax.plot(range(len(true_rul)), true_rul, "k-", lw=1.5, alpha=0.8, label="真实RUL")
            for name in show_models:
                key = model_map[name]
                pred = coupled_pred if key == "coupled" else single_preds.get(key, [])
                if len(pred) > 0:
                    ls = "-" if key == "coupled" else "--"
                    ax.plot(range(len(pred)), pred, ls, color=colors[name],
                            alpha=0.7, lw=1.5, label=name)
            ax.set_xlabel("发动机编号")
            ax.set_ylabel("剩余使用寿命 (RUL)")
            ax.set_title("测试集 RUL 预测对比")
        else:
            for name in show_models:
                key = model_map[name]
                pred = coupled_pred if key == "coupled" else single_preds.get(key, [])
                if len(pred) > 0:
                    ax.scatter(true_rul, pred, c=colors[name], alpha=0.4,
                               s=25, label=name, edgecolors="none")
            mx = max(true_rul.max(), 130)
            ax.plot([0, mx], [0, mx], "k--", alpha=0.5, label="理想预测线")
            ax.fill_between([0, mx], [-20, mx-20], [20, mx+20],
                            alpha=0.06, color="green", label="±20区间")
            ax.set_xlabel("真实RUL")
            ax.set_ylabel("预测RUL")
            ax.set_title("预测精度散点图")
            ax.set_xlim(0, mx)
            ax.set_ylim(0, mx)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()


def render_monte_carlo_tab(pipe):
    st.subheader("蒙特卡洛寿命仿真")
    st.markdown("调节应力倍率，通过蒙特卡洛采样观察系统寿命分布变化。")

    col1, col2 = st.columns([1, 3])
    with col1:
        temp_mult = st.slider("温度应力倍率", 0.5, 3.0, 1.0, 0.1)
        mech_mult = st.slider("机械应力倍率", 0.5, 3.0, 1.0, 0.1)
        therm_mult = st.slider("热负荷应力倍率", 0.5, 3.0, 1.0, 0.1)
        n_samples = st.select_slider("采样次数", [100, 500, 1000, 5000, 10000], value=1000)

    mults = {"temperature": temp_mult, "mechanical": mech_mult, "thermal": therm_mult}
    samples = pipe.monte_carlo_lifetime(n_samples=n_samples, stress_multipliers=mults)

    with col2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.hist(samples, bins=50, color="#8e44ad", alpha=0.7, edgecolor="white", density=True)
        ax.axvline(np.mean(samples), color="red", ls="--", lw=2,
                   label=f"均值={np.mean(samples):.0f}")
        ax.axvline(np.median(samples), color="orange", ls=":", lw=2,
                   label=f"中位数={np.median(samples):.0f}")
        ax.set_xlabel("系统寿命（周期）")
        ax.set_ylabel("概率密度")
        ax.set_title(f"蒙特卡洛寿命分布（N={n_samples}）")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        sorted_s = np.sort(samples)
        reliability = 1.0 - np.arange(1, len(sorted_s)+1) / len(sorted_s)
        ax.plot(sorted_s, reliability, "b-", lw=2)
        ax.axhline(0.8, color="red", ls=":", alpha=0.5, label="R=0.8阈值")
        idx_80 = np.searchsorted(-reliability, -0.8)
        if idx_80 < len(sorted_s):
            ax.axvline(sorted_s[idx_80], color="green", ls="--", alpha=0.7,
                       label=f"R=0.8对应寿命≈{sorted_s[idx_80]:.0f}")
        ax.set_xlabel("运行周期")
        ax.set_ylabel("可靠度 R(t)")
        ax.set_title("经验可靠度曲线")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("平均寿命", f"{np.mean(samples):.0f} 周期")
    c2.metric("标准差", f"{np.std(samples):.0f} 周期")
    c3.metric("5%分位数", f"{np.percentile(samples, 5):.0f} 周期")
    c4.metric("综合应力倍率", f"{np.mean(list(mults.values())):.2f}x")


def render_metrics_tab(pipe):
    st.subheader("模型评估指标汇总")

    true_rul, single_preds, coupled_pred = pipe.get_test_predictions()
    import pandas as pd

    rows = []
    for key, pred in single_preds.items():
        m = evaluate(true_rul, pred)
        m["模型"] = STRESS_CN[key]
        rows.append(m)
    m = evaluate(true_rul, coupled_pred)
    m["模型"] = "多应力耦合"
    rows.append(m)

    df = pd.DataFrame(rows).set_index("模型")
    df = df[["RMSE", "MAE", "R2", "Acc20", "PHM08"]]
    df.columns = ["RMSE", "MAE", "R²", "±20准确率(%)", "PHM08得分"]

    st.dataframe(df.style.format({
        "RMSE": "{:.2f}", "MAE": "{:.2f}", "R²": "{:.4f}",
        "±20准确率(%)": "{:.1f}", "PHM08得分": "{:.0f}"
    }).highlight_min(subset=["RMSE", "MAE", "PHM08得分"], color="#d4edda")
     .highlight_max(subset=["R²", "±20准确率(%)"], color="#d4edda"),
     width="stretch")

    st.markdown("---")
    st.markdown("**指标说明**")
    st.markdown("""
    - **RMSE**: 均方根误差，越低越好
    - **MAE**: 平均绝对误差，越低越好
    - **R²**: 决定系数，越接近1越好
    - **±20准确率**: 预测误差在±20周期内的比例
    - **PHM08得分**: PHM08竞赛评分，越低越好（对晚预测惩罚更重）
    """)

    sup = np.mean([np.array(v) for v in single_preds.values()], axis=0)
    eff = np.array(coupled_pred) - sup
    synergy = np.mean(eff < 0) * 100
    antagonism = np.mean(eff > 0) * 100

    st.markdown("---")
    st.subheader("非线性耦合效应分析")
    c1, c2 = st.columns(2)
    c1.metric("协同加速效应占比", f"{synergy:.0f}%")
    c2.metric("拮抗效应占比", f"{antagonism:.0f}%")


if __name__ == "__main__":
    main()
