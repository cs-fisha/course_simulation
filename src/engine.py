"""
多环境应力耦合失效机理仿真引擎
基于 NASA C-MAPSS FD004 数据集
技术路线: LightGBM + Weibull + HMM + SHAP
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")

COLUMN_NAMES = (
    ["unit", "cycle"]
    + ["os_altitude", "os_mach", "os_tra"]
    + [f"s_{i}" for i in range(1, 22)]
)
RUL_CLIP = 125
WINDOW = 15
KEY_SENSORS = ["s_3", "s_4", "s_7", "s_8", "s_9", "s_11", "s_12",
               "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"]

STRESS_CN = {
    "altitude": "温度应力(高度)",
    "mach": "机械应力(马赫数)",
    "tra": "热负荷应力(TRA)",
}


def load_fd004():
    train_path = os.path.join(DATA_DIR, "train_FD004.txt")
    test_path = os.path.join(DATA_DIR, "test_FD004.txt")
    rul_path = os.path.join(DATA_DIR, "RUL_FD004.txt")
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    rul_true = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])
    mc = train_df.groupby("unit")["cycle"].max().reset_index()
    mc.columns = ["unit", "max_cycle"]
    train_df = train_df.merge(mc, on="unit")
    train_df["RUL"] = (train_df["max_cycle"] - train_df["cycle"]).clip(upper=RUL_CLIP)
    train_df.drop("max_cycle", axis=1, inplace=True)
    return train_df, test_df, rul_true


def preprocess(train_df, test_df):
    sensor_cols = [f"s_{i}" for i in range(1, 22)]
    valid_sensors = [c for c in sensor_cols if train_df[c].std() > 1e-6]

    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    train_df["oc"] = kmeans.fit_predict(train_df[["os_altitude", "os_mach", "os_tra"]])
    test_df["oc"] = kmeans.predict(test_df[["os_altitude", "os_mach", "os_tra"]])

    oc_stats = {}
    for col in valid_sensors:
        grp = train_df.groupby("oc")[col].agg(["mean", "std"])
        grp["std"] = grp["std"].replace(0, 1)
        oc_stats[col] = grp
        train_df[f"{col}_n"] = (train_df[col] - train_df["oc"].map(grp["mean"])) / train_df["oc"].map(grp["std"])
        test_df[f"{col}_n"] = (test_df[col] - test_df["oc"].map(grp["mean"]).fillna(0)) / test_df["oc"].map(grp["std"]).fillna(1)

    norm_cols = [f"{c}_n" for c in valid_sensors]
    for df in [train_df, test_df]:
        grouped = df.groupby("unit")
        for col in norm_cols:
            rolled = grouped[col].rolling(WINDOW, min_periods=1)
            df[f"{col}_ma"] = rolled.mean().droplevel(0)
            df[f"{col}_sd"] = rolled.std().droplevel(0).fillna(0)

    ks = [s for s in KEY_SENSORS if s in valid_sensors]
    ma_cols_hi = [f"{s}_n_ma" for s in ks]
    for df in [train_df, test_df]:
        vals = df[ma_cols_hi].values
        df["HI"] = np.mean(np.abs(vals), axis=1)
    hi_lo = train_df["HI"].quantile(0.02)
    hi_hi = train_df["HI"].quantile(0.98)
    hi_rng = max(hi_hi - hi_lo, 1e-6)
    for df in [train_df, test_df]:
        df["HI"] = np.clip((df["HI"] - hi_lo) / hi_rng, 0, 1)

    max_cycle_typical = train_df.groupby("unit")["cycle"].max().median()
    for df in [train_df, test_df]:
        df["cycle_n"] = df["cycle"] / max_cycle_typical

    slope_window = 10
    for df in [train_df, test_df]:
        grouped = df.groupby("unit")
        for s in ks:
            col_ma = f"{s}_n_ma"
            if col_ma in df.columns:
                rolled = grouped[col_ma].diff(slope_window)
                df[f"{s}_slope"] = rolled.fillna(0) / slope_window

    cumdev_refs = {}
    for s in ks:
        col_ma = f"{s}_n_ma"
        if col_ma in train_df.columns:
            unit_totals = train_df.groupby("unit")[col_ma].apply(lambda x: x.abs().sum())
            cumdev_refs[s] = unit_totals.median()

    for df in [train_df, test_df]:
        grouped = df.groupby("unit")
        for s in ks:
            col_ma = f"{s}_n_ma"
            if col_ma in df.columns:
                ref = max(cumdev_refs.get(s, 1.0), 1e-6)
                df[f"{s}_cumdev"] = grouped[col_ma].cumsum() / ref

    return train_df, test_df, valid_sensors, kmeans, oc_stats


def build_coupled_features(df, valid_sensors, stress_scaler=None):
    ma_cols = [f"{c}_n_ma" for c in valid_sensors if f"{c}_n_ma" in df.columns]
    sd_cols = [f"{c}_n_sd" for c in valid_sensors if f"{c}_n_sd" in df.columns]

    stress_raw = ["os_altitude", "os_mach", "os_tra"]
    if stress_scaler is None:
        stress_scaler = MinMaxScaler()
        sn = stress_scaler.fit_transform(df[stress_raw])
    else:
        sn = stress_scaler.transform(df[stress_raw])

    df["alt_n"] = sn[:, 0]
    df["mach_n"] = sn[:, 1]
    df["tra_n"] = sn[:, 2]
    df["alt_x_mach"] = df["alt_n"] * df["mach_n"]
    df["mach_x_tra"] = df["mach_n"] * df["tra_n"]
    df["alt_x_tra"] = df["alt_n"] * df["tra_n"]
    df["triple"] = df["alt_n"] * df["mach_n"] * df["tra_n"]

    for s in KEY_SENSORS:
        if f"{s}_n_ma" in df.columns:
            df[f"{s}_xa"] = df[f"{s}_n_ma"] * df["alt_n"]
            df[f"{s}_xm"] = df[f"{s}_n_ma"] * df["mach_n"]

    interact = ["alt_n", "mach_n", "tra_n", "alt_x_mach", "mach_x_tra", "alt_x_tra", "triple"]
    cross_cols = [c for c in df.columns if c.endswith("_xa") or c.endswith("_xm")]
    slope_cols = [c for c in df.columns if c.endswith("_slope")]
    cumdev_cols = [c for c in df.columns if c.endswith("_cumdev")]

    feat_cols = ma_cols + sd_cols + interact + cross_cols + slope_cols + cumdev_cols
    if "HI" in df.columns:
        feat_cols.append("HI")
    if "cycle_n" in df.columns:
        feat_cols.append("cycle_n")
    feat_cols = [c for c in feat_cols if c in df.columns]
    return df, feat_cols, stress_scaler


def build_single_features(df, valid_sensors, target_stress):
    sensor_groups = {
        "os_altitude": ["s_3", "s_4", "s_7", "s_11", "s_12", "s_15"],
        "os_mach":     ["s_8", "s_9", "s_13", "s_14", "s_17", "s_21"],
        "os_tra":      ["s_3", "s_8", "s_9", "s_11", "s_14", "s_17"],
    }
    group = [s for s in sensor_groups.get(target_stress, valid_sensors) if s in valid_sensors]
    return [f"{c}_n_ma" for c in group if f"{c}_n_ma" in df.columns]


LGB_NUM_THREADS = int(os.environ.get("LGBM_NUM_THREADS", "4"))

LGB_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1,
    min_child_samples=15, verbose=-1, n_jobs=LGB_NUM_THREADS, random_state=42
)


class SingleStressModel:
    def __init__(self, name, col):
        self.name = name
        self.col = col
        self.model = None
        self.feat_cols = None
        self.weibull_params = None
        self.weibull_by_level = {}

    def fit(self, train_df, feat_cols):
        self.feat_cols = feat_cols
        X = train_df[feat_cols].values
        y = train_df["RUL"].values
        self.model = lgb.LGBMRegressor(**LGB_PARAMS)
        self.model.fit(X, y)

        all_lives = train_df.groupby("unit")["cycle"].max().values.astype(float)
        self.weibull_params = weibull_min.fit(all_lives, floc=0)

        unit_exp = train_df.groupby("unit").agg(
            stress_mean=(self.col, "mean"), life=("cycle", "max"))
        p33, p67 = np.percentile(unit_exp["stress_mean"], [33, 67])
        for lbl, m in [("低", unit_exp["stress_mean"] <= p33),
                       ("中", (unit_exp["stress_mean"] > p33) & (unit_exp["stress_mean"] <= p67)),
                       ("高", unit_exp["stress_mean"] > p67)]:
            sub = unit_exp[m]
            if len(sub) >= 5:
                try:
                    self.weibull_by_level[lbl] = weibull_min.fit(sub["life"].values.astype(float), floc=0)
                except Exception:
                    pass

    def predict(self, df):
        X = df[self.feat_cols].fillna(0).values
        return np.clip(self.model.predict(X), 0, RUL_CLIP)

    def reliability(self, t):
        sh, lc, sc = self.weibull_params
        return 1.0 - weibull_min.cdf(t, sh, loc=lc, scale=sc)

    def mean_life(self):
        sh, lc, sc = self.weibull_params
        return weibull_min.mean(sh, loc=lc, scale=sc)


class CoupledModel:
    def __init__(self):
        self.model = None
        self.feat_cols = None

    def fit(self, df, feat_cols):
        self.feat_cols = feat_cols
        X = df[feat_cols].values
        y = df["RUL"].values
        self.model = lgb.LGBMRegressor(
            n_estimators=500, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.05,
            min_child_samples=10, verbose=-1, n_jobs=LGB_NUM_THREADS, random_state=42)
        self.model.fit(X, y)

    def predict(self, df):
        X = df[self.feat_cols].fillna(0).values
        return np.clip(self.model.predict(X), 0, RUL_CLIP)


class DegradationHMM:
    N_STATES = 5
    LABELS = ["健康", "轻微退化", "中度退化", "严重退化", "失效临界"]

    def __init__(self, n_obs=10):
        self.n_obs = n_obs
        self.pi = np.array([0.85, 0.10, 0.04, 0.01, 0.0])
        self.A = np.array([
            [0.94, 0.05, 0.01, 0.00, 0.00],
            [0.00, 0.91, 0.07, 0.02, 0.00],
            [0.00, 0.00, 0.87, 0.10, 0.03],
            [0.00, 0.00, 0.00, 0.80, 0.20],
            [0.00, 0.00, 0.00, 0.00, 1.00]])
        self.B = np.zeros((self.N_STATES, n_obs))
        for i in range(self.N_STATES):
            ctr = i * (n_obs - 1) / (self.N_STATES - 1)
            for j in range(n_obs):
                self.B[i, j] = np.exp(-0.5 * ((j - ctr) / max(1, n_obs / 7)) ** 2)
            self.B[i] /= self.B[i].sum()

    def discretize(self, vals):
        return np.floor(np.clip(vals, 0, 0.999) * self.n_obs).astype(int).clip(0, self.n_obs - 1)

    def _forward(self, obs):
        T = len(obs)
        a = np.zeros((T, self.N_STATES))
        c = np.zeros(T)
        a[0] = self.pi * self.B[:, obs[0]]
        c[0] = a[0].sum() or 1e-300
        a[0] /= c[0]
        for t in range(1, T):
            a[t] = (a[t-1] @ self.A) * self.B[:, obs[t]]
            c[t] = a[t].sum() or 1e-300
            a[t] /= c[t]
        return a, c

    def _backward(self, obs, c):
        T = len(obs)
        b = np.zeros((T, self.N_STATES))
        b[-1] = 1.0
        for t in range(T-2, -1, -1):
            b[t] = self.A @ (self.B[:, obs[t+1]] * b[t+1])
            b[t] /= (c[t+1] or 1e-300)
        return b

    def baum_welch(self, seqs, max_iter=50, tol=1e-4):
        for it in range(max_iter):
            An = np.zeros_like(self.A)
            Ad = np.zeros(self.N_STATES)
            Bn = np.zeros_like(self.B)
            Bd = np.zeros(self.N_STATES)
            for obs in seqs:
                if len(obs) < 3:
                    continue
                a, c = self._forward(obs)
                b = self._backward(obs, c)
                g = a * b
                g /= (g.sum(axis=1, keepdims=True) + 1e-300)
                for t in range(len(obs)-1):
                    xi = np.outer(a[t], b[t+1]) * self.A * self.B[:, obs[t+1]]
                    xi /= (xi.sum() or 1e-300)
                    An += xi
                    Ad += g[t]
                for t in range(len(obs)):
                    Bn[:, obs[t]] += g[t]
                    Bd += g[t]
            nA = An / np.maximum(Ad, 1e-300)[:, None]
            for i in range(self.N_STATES):
                nA[i, :i] = 0
                s = nA[i].sum()
                if s > 0: nA[i] /= s
            nB = Bn / np.maximum(Bd, 1e-300)[:, None]
            for i in range(self.N_STATES):
                s = nB[i].sum()
                if s > 0: nB[i] /= s
            d = np.max(np.abs(nA - self.A))
            self.A, self.B = nA, nB
            if d < tol:
                break

    def viterbi(self, obs):
        T = len(obs)
        lA, lB = np.log(self.A + 1e-300), np.log(self.B + 1e-300)
        d = np.zeros((T, self.N_STATES))
        psi = np.zeros((T, self.N_STATES), dtype=int)
        d[0] = np.log(self.pi + 1e-300) + lB[:, obs[0]]
        for t in range(1, T):
            for j in range(self.N_STATES):
                cand = d[t-1] + lA[:, j]
                psi[t, j] = np.argmax(cand)
                d[t, j] = cand[psi[t, j]] + lB[j, obs[t]]
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(d[-1])
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        return path

    def forward_predict(self, obs, steps=100):
        a, _ = self._forward(obs)
        p = a[-1] / (a[-1].sum() + 1e-300)
        preds = np.zeros((steps, self.N_STATES))
        for s in range(steps):
            p = p @ self.A
            preds[s] = p
        return preds


def phm08_score(true, pred):
    d = np.array(pred) - np.array(true)
    return np.sum(np.where(d < 0, np.exp(-d/13) - 1, np.exp(d/10) - 1))


def evaluate(true, pred):
    t, p = np.array(true, dtype=float), np.clip(np.array(pred, dtype=float), 0, RUL_CLIP)
    mae = mean_absolute_error(t, p)
    rmse = np.sqrt(mean_squared_error(t, p))
    r2 = 1 - np.sum((t-p)**2) / (np.sum((t-t.mean())**2) + 1e-10)
    acc20 = np.mean(np.abs(t-p) <= 20) * 100
    sc = phm08_score(t, p)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Acc20": acc20, "PHM08": sc}


class SimulationPipeline:
    """完整仿真流水线，训练后可供 Streamlit 交互调用"""

    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.rul_true = None
        self.valid_sensors = None
        self.kmeans = None
        self.stress_scaler = None
        self.coupled_feats = None
        self.single_models = {}
        self.coupled_model = None
        self.hmm = None
        self._trained = False

    def load_and_train(self):
        self.train_df, self.test_df, self.rul_true = load_fd004()
        self.train_df, self.test_df, self.valid_sensors, self.kmeans, _ = preprocess(
            self.train_df, self.test_df)
        self.train_df, self.coupled_feats, self.stress_scaler = build_coupled_features(
            self.train_df, self.valid_sensors)
        self.test_df, _, _ = build_coupled_features(
            self.test_df, self.valid_sensors, self.stress_scaler)

        stress_config = {"altitude": "os_altitude", "mach": "os_mach", "tra": "os_tra"}
        for key, col in stress_config.items():
            mdl = SingleStressModel(key, col)
            feat_cols = build_single_features(self.train_df, self.valid_sensors, col)
            mdl.fit(self.train_df, feat_cols)
            self.single_models[key] = mdl

        self.coupled_model = CoupledModel()
        self.coupled_model.fit(self.train_df, self.coupled_feats)

        self.hmm = DegradationHMM(n_obs=10)
        self._train_hmm()
        self._trained = True

    def _train_hmm(self):
        ks = [s for s in KEY_SENSORS if s in self.valid_sensors]
        ma_cols = [f"{s}_n_ma" for s in ks if f"{s}_n_ma" in self.train_df.columns]
        rng = np.random.RandomState(42)
        seqs = []
        for uid in sorted(self.train_df["unit"].unique())[:50]:
            sub = self.train_df[self.train_df["unit"] == uid]
            rul_vals = sub["RUL"].values
            max_r = max(rul_vals.max(), 1)
            deg = 1.0 - rul_vals / max_r
            if ma_cols:
                sensor_deg = np.mean(np.abs(sub[ma_cols].values), axis=1)
                lo, hi = np.percentile(sensor_deg, [5, 95])
                sensor_n = np.clip((sensor_deg - lo) / max(hi - lo, 1), 0, 1)
                deg = 0.6 * deg + 0.4 * sensor_n
            deg += rng.normal(0, 0.02, len(deg))
            seqs.append(self.hmm.discretize(np.clip(deg, 0, 0.999)))
        self.hmm.baum_welch(seqs, max_iter=50)
        self._hmm_seqs = seqs

    def get_test_predictions(self):
        test_last = self.test_df.groupby("unit").last().reset_index()
        n_eval = min(len(self.rul_true), len(test_last))
        true_rul = self.rul_true["RUL"].values[:n_eval].clip(0, RUL_CLIP)
        test_last = test_last.iloc[:n_eval].copy()

        single_preds = {}
        for key, mdl in self.single_models.items():
            single_preds[key] = mdl.predict(test_last)
        coupled_pred = self.coupled_model.predict(test_last)
        return true_rul, single_preds, coupled_pred

    def get_unit_trajectory(self, unit_id):
        sub = self.train_df[self.train_df["unit"] == unit_id].copy()
        if len(sub) == 0:
            return None
        ks = [s for s in KEY_SENSORS if s in self.valid_sensors]
        ma_cols = [f"{s}_n_ma" for s in ks if f"{s}_n_ma" in sub.columns]
        rul_vals = sub["RUL"].values
        max_r = max(rul_vals.max(), 1)
        deg = 1.0 - rul_vals / max_r
        if ma_cols:
            sensor_deg = np.mean(np.abs(sub[ma_cols].values), axis=1)
            lo, hi = np.percentile(sensor_deg, [5, 95])
            sensor_n = np.clip((sensor_deg - lo) / max(hi - lo, 1), 0, 1)
            deg = 0.6 * deg + 0.4 * sensor_n
        obs = self.hmm.discretize(np.clip(deg, 0, 0.999))
        path = self.hmm.viterbi(obs)
        return {
            "cycles": sub["cycle"].values,
            "rul": rul_vals,
            "hi": sub["HI"].values,
            "deg": deg,
            "hmm_states": path,
            "obs": obs,
        }

    def weibull_reliability_curve(self, stress_key, t_range=None):
        if t_range is None:
            t_range = np.linspace(1, 500, 500)
        mdl = self.single_models[stress_key]
        result = {"t": t_range, "overall": mdl.reliability(t_range)}
        for lbl, params in mdl.weibull_by_level.items():
            sh, lc, sc = params
            result[lbl] = 1.0 - weibull_min.cdf(t_range, sh, loc=lc, scale=sc)
        return result

    def monte_carlo_lifetime(self, n_samples=1000, stress_multipliers=None):
        """蒙特卡洛寿命仿真：给定应力倍率，采样寿命分布"""
        sh, lc, sc = self.single_models["altitude"].weibull_params
        if stress_multipliers is None:
            stress_multipliers = {"temperature": 1.0, "mechanical": 1.0, "thermal": 1.0}
        avg_mult = np.mean(list(stress_multipliers.values()))
        adjusted_scale = sc / max(avg_mult, 0.1)
        samples = weibull_min.rvs(sh, loc=lc, scale=adjusted_scale, size=n_samples,
                                  random_state=42)
        return samples.clip(10, 1000)

    @property
    def unit_ids(self):
        if self.train_df is not None:
            return sorted(self.train_df["unit"].unique())
        return []
