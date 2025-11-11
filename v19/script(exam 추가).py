# script.py (v19, exam-order aware + Delta+log_ratio, 5-fold CatBoost + Isotonic)
# -------------------------------------------------------------------------
# Inference for:
#  - A: Base + Hist + Norm + log_ratio (PK stats 미사용, Delta 미사용)
#  - B: Base + PK Stats + Hist + Norm + Delta(B-only) + log_ratio
#  - Exam-order aware features: is_first_in_test, exam_order_bucket, prior_risk_by_order
#  - 5-fold CatBoost + Isotonic calibration ensemble
#  - Uses models & stats saved in ./model
#  - Reads test data from ./data and writes ./output/submission.csv
# -------------------------------------------------------------------------

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import catboost as cb  # noqa: F401 (for type compatibility)

# -----------------------------
# 1. 경로 및 상수
# -----------------------------
BASE_DIR = "./data"
MODEL_SAVE_DIR = "./model"
OUTPUT_DIR = "./output"
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

N_SPLITS = 5
# 학습 시 사용한 범주형 정의(정확히 일치해야 함)
CAT_FEATURES = ["Age", "PrimaryKey", "pk_hist_total_count", "exam_order_bucket", "is_first_in_test"]

# EDA 기반 기본 prior (order_prior.csv가 없을 때 사용)
DEFAULT_ORDER_PRIOR = {
    "A_1": 0.021556,
    "A_2": 0.041630,
    "A_3plus": 0.04934791076451036,
    "B_1": 0.040935,
    "B_2to3": 0.04351909134533335,
    "B_4plus": 0.05579803003003003,
}
DEFAULT_GLOBAL_PRIOR = 0.04

# -----------------------------
# 2. 공용 유틸(도메인 전처리)
# -----------------------------
def convert_age(val):
    if pd.isna(val):
        return np.nan
    try:
        s = str(val)
        base = int(s[:-1])
        return base if s[-1] == "a" else base + 5
    except Exception:
        return np.nan

def split_testdate(val):
    try:
        v = int(val)
        return v // 100, v % 100
    except Exception:
        return np.nan, np.nan

def seq_mean(series: pd.Series) -> pd.Series:
    return series.fillna("").apply(
        lambda x: np.fromstring(x, sep=",").mean() if x else np.nan
    )

def seq_std(series: pd.Series) -> pd.Series:
    return series.fillna("").apply(
        lambda x: np.fromstring(x, sep=",").std() if x else np.nan
    )

def masked_operation(cond_series, val_series, target_conds, operation="mean"):
    cond_df = (
        cond_series.fillna("")
        .str.split(",", expand=True)
        .replace("", np.nan)
        .to_numpy(dtype=float)
    )
    val_df = (
        val_series.fillna("")
        .str.split(",", expand=True)
        .replace("", np.nan)
        .to_numpy(dtype=float)
    )

    if isinstance(target_conds, (list, set, tuple)):
        mask = np.isin(cond_df, list(target_conds))
    else:
        mask = cond_df == target_conds

    masked_vals = np.where(mask, val_df, np.nan)

    with np.errstate(invalid="ignore"):
        if operation == "mean":
            sums = np.nansum(masked_vals, axis=1)
            counts = np.sum(mask, axis=1)
            out = sums / np.where(counts == 0, np.nan, counts)
        elif operation == "std":
            out = np.nanstd(masked_vals, axis=1)
        elif operation == "rate":
            corrects = np.nansum(np.where(masked_vals == 1, 1, 0), axis=1)
            total = np.sum(mask, axis=1)
            out = corrects / np.where(total == 0, np.nan, total)
        else:
            out = np.nan
    return pd.Series(out, index=cond_series.index)

# ---- PDF 기반 정답률 계산 함수들 ----
def seq_rate_A3(series, target_codes):
    def calc(x):
        if not x:
            return np.nan
        s = x.split(",")
        correct = sum(s.count(code) for code in target_codes if code in ["1", "3"])
        incorrect = sum(s.count(code) for code in target_codes if code in ["2", "4"])
        total = correct + incorrect
        return correct / total if total > 0 else np.nan
    return series.fillna("").apply(calc)

def seq_rate_B1_B2(series, target_codes):
    def calc(x):
        if not x:
            return np.nan
        s = x.split(",")
        correct = sum(s.count(code) for code in target_codes if code in ["1", "3"])
        incorrect = sum(s.count(code) for code in target_codes if code in ["2", "4"])
        total = correct + incorrect
        return correct / total if total > 0 else np.nan
    return series.fillna("").apply(calc)

def seq_rate_B4(series, target_codes):
    def calc(x):
        if not x:
            return np.nan
        s = x.split(",")
        correct = sum(s.count(code) for code in target_codes if code in ["1", "3", "5"])
        incorrect = sum(s.count(code) for code in target_codes if code in ["2", "4", "6"])
        total = correct + incorrect
        return correct / total if total > 0 else np.nan
    return series.fillna("").apply(calc)

def seq_rate_simple(series):  # B3, B5, B6, B7, B8
    def calc(x):
        if not x:
            return np.nan
        s = x.split(",")
        correct = s.count("1")
        incorrect = s.count("2")
        total = correct + incorrect
        return correct / total if total > 0 else np.nan
    return series.fillna("").apply(calc)

# -----------------------------
# 3. 1차 Feature Engineering (A/B)
# -----------------------------
def preprocess_A(df):
    df = df.copy()
    df["Age"] = df["Age"]
    df["Age_num"] = df["Age"].map(convert_age)
    ym = df["TestDate"].map(split_testdate)
    df["Year"] = [y for y, m in ym]
    df["Month"] = [m for y, m in ym]

    feats = pd.DataFrame(index=df.index)

    # A1
    feats["A1_rt_mean"] = seq_mean(df["A1-4"])
    feats["A1_rt_std"] = seq_std(df["A1-4"])
    feats["A1_rt_left"] = masked_operation(df["A1-1"], df["A1-4"], 1, "mean")
    feats["A1_rt_right"] = masked_operation(df["A1-1"], df["A1-4"], 2, "mean")
    feats["A1_rt_slow"] = masked_operation(df["A1-2"], df["A1-4"], 1, "mean")
    feats["A1_rt_norm"] = masked_operation(df["A1-2"], df["A1-4"], 2, "mean")
    feats["A1_rt_fast"] = masked_operation(df["A1-2"], df["A1-4"], 3, "mean")
    feats["A1_acc_slow"] = masked_operation(df["A1-2"], df["A1-3"], 1, "rate")
    feats["A1_acc_norm"] = masked_operation(df["A1-2"], df["A1-3"], 2, "rate")
    feats["A1_acc_fast"] = masked_operation(df["A1-2"], df["A1-3"], 3, "rate")

    # A2
    feats["A2_rt_mean"] = seq_mean(df["A2-4"])
    feats["A2_rt_std"] = seq_std(df["A2-4"])
    feats["A2_rt_slow_c1"] = masked_operation(df["A2-1"], df["A2-4"], 1, "mean")
    feats["A2_rt_norm_c1"] = masked_operation(df["A2-1"], df["A2-4"], 2, "mean")
    feats["A2_rt_fast_c1"] = masked_operation(df["A2-1"], df["A2-4"], 3, "mean")
    feats["A2_rt_slow_c2"] = masked_operation(df["A2-2"], df["A2-4"], 1, "mean")
    feats["A2_rt_norm_c2"] = masked_operation(df["A2-2"], df["A2-4"], 2, "mean")
    feats["A2_rt_fast_c2"] = masked_operation(df["A2-2"], df["A2-4"], 3, "mean")
    feats["A2_acc_slow"] = masked_operation(df["A2-1"], df["A2-3"], 1, "rate")
    feats["A2_acc_norm"] = masked_operation(df["A2-1"], df["A2-3"], 2, "rate")
    feats["A2_acc_fast"] = masked_operation(df["A2-1"], df["A2-3"], 3, "rate")

    # A3
    feats["A3_valid_acc"] = seq_rate_A3(df["A3-5"], ["1", "2"])
    feats["A3_invalid_acc"] = seq_rate_A3(df["A3-5"], ["3", "4"])
    feats["A3_rt_mean"] = seq_mean(df["A3-7"])
    feats["A3_rt_std"] = seq_std(df["A3-7"])
    feats["A3_rt_small"] = masked_operation(df["A3-1"], df["A3-7"], 1, "mean")
    feats["A3_rt_big"] = masked_operation(df["A3-1"], df["A3-7"], 2, "mean")
    feats["A3_rt_left"] = masked_operation(df["A3-3"], df["A3-7"], 1, "mean")
    feats["A3_rt_right"] = masked_operation(df["A3-3"], df["A3-7"], 2, "mean")

    # A4
    feats["A4_rt_mean"] = seq_mean(df["A4-5"])
    feats["A4_rt_std"] = seq_std(df["A4-5"])
    feats["A4_rt_congruent"] = masked_operation(df["A4-1"], df["A4-5"], 1, "mean")
    feats["A4_rt_incongruent"] = masked_operation(df["A4-1"], df["A4-5"], 2, "mean")
    feats["A4_acc_congruent"] = masked_operation(df["A4-1"], df["A4-3"], 1, "rate")
    feats["A4_acc_incongruent"] = masked_operation(df["A4-1"], df["A4-3"], 2, "rate")

    # A5
    feats["A5_acc_nonchange"] = masked_operation(df["A5-1"], df["A5-2"], 1, "rate")
    feats["A5_acc_pos_change"] = masked_operation(df["A5-1"], df["A5-2"], 2, "rate")
    feats["A5_acc_color_change"] = masked_operation(df["A5-1"], df["A5-2"], 3, "rate")
    feats["A5_acc_shape_change"] = masked_operation(df["A5-1"], df["A5-2"], 4, "rate")

    # A6, A7
    feats["A6_correct_count"] = df["A6-1"]
    feats["A7_correct_count"] = df["A7-1"]

    # A8, A9
    feats["A8-1"] = df["A8-1"]; feats["A8-2"] = df["A8-2"]
    feats["A9-1"] = df["A9-1"]; feats["A9-2"] = df["A9-2"]; feats["A9-3"] = df["A9-3"]; feats["A9-4"] = df["A9-4"]; feats["A9-5"] = df["A9-5"]

    seq_cols = [
        "A1-1","A1-2","A1-3","A1-4",
        "A2-1","A2-2","A2-3","A2-4",
        "A3-1","A3-2","A3-3","A3-4","A3-5","A3-6","A3-7",
        "A4-1","A4-2","A4-3","A4-4","A4-5",
        "A5-1","A5-2","A5-3",
        "A6-1","A7-1","A8-1","A8-2",
        "A9-1","A9-2","A9-3","A9-4","A9-5",
    ]
    out = pd.concat([df, feats], axis=1).drop(columns=seq_cols, errors="ignore")
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out

def preprocess_B(df):
    df = df.copy()
    df["Age"] = df["Age"]
    df["Age_num"] = df["Age"].map(convert_age)
    ym = df["TestDate"].map(split_testdate)
    df["Year"] = [y for y, m in ym]
    df["Month"] = [m for y, m in ym]

    feats = pd.DataFrame(index=df.index)

    # B1, B2
    feats["B1_task1_acc"] = seq_rate_simple(df["B1-1"])
    feats["B1_rt_mean"] = seq_mean(df["B1-2"])
    feats["B1_rt_std"] = seq_std(df["B1-2"])
    feats["B1_change_acc"] = seq_rate_B1_B2(df["B1-3"], ["1", "2"])
    feats["B1_nonchange_acc"] = seq_rate_B1_B2(df["B1-3"], ["3", "4"])

    feats["B2_task1_acc"] = seq_rate_simple(df["B2-1"])
    feats["B2_rt_mean"] = seq_mean(df["B2-2"])
    feats["B2_rt_std"] = seq_std(df["B2-2"])
    feats["B2_change_acc"] = seq_rate_B1_B2(df["B2-3"], ["1", "2"])
    feats["B2_nonchange_acc"] = seq_rate_B1_B2(df["B2-3"], ["3", "4"])

    # B3
    feats["B3_acc_rate"] = seq_rate_simple(df["B3-1"])
    feats["B3_rt_mean"] = seq_mean(df["B3-2"])
    feats["B3_rt_std"] = seq_std(df["B3-2"])

    # B4
    feats["B4_congruent_acc"] = seq_rate_B4(df["B4-1"], ["1", "2"])
    feats["B4_incongruent_acc"] = seq_rate_B4(df["B4-1"], ["3", "4", "5", "6"])
    feats["B4_rt_mean"] = seq_mean(df["B4-2"])
    feats["B4_rt_std"] = seq_std(df["B4-2"])

    # B5~B8
    feats["B5_acc_rate"] = seq_rate_simple(df["B5-1"])
    feats["B5_rt_mean"] = seq_mean(df["B5-2"])
    feats["B5_rt_std"] = seq_std(df["B5-2"])
    feats["B6_acc_rate"] = seq_rate_simple(df["B6"])
    feats["B7_acc_rate"] = seq_rate_simple(df["B7"])
    feats["B8_acc_rate"] = seq_rate_simple(df["B8"])

    # B9, B10
    for c in ["B9-1","B9-2","B9-3","B9-4","B9-5"]:
        feats[c] = df[c]
    for c in ["B10-1","B10-2","B10-3","B10-4","B10-5","B10-6"]:
        feats[c] = df[c]

    seq_cols = [
        "B1-1","B1-2","B1-3",
        "B2-1","B2-2","B2-3",
        "B3-1","B3-2",
        "B4-1","B4-2",
        "B5-1","B5-2",
        "B6","B7","B8",
        "B9-1","B9-2","B9-3","B9-4","B9-5",
        "B10-1","B10-2","B10-3","B10-4","B10-5","B10-6",
    ]
    out = pd.concat([df, feats], axis=1).drop(columns=seq_cols, errors="ignore")
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out

# -----------------------------
# 4. 2차 Feature Engineering (log_ratio 등)
# -----------------------------
def _has(df, cols):
    return all(c in df.columns for c in cols)

def _safe_div(a, b, eps=1e-6):
    return a / (b + eps)

def _log_ratio(num, den, eps=1e-6):
    return np.log((num + eps) / (den + eps))

def add_features_A(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy()
    eps = 1e-6
    if _has(feats, ["Year","Month"]):
        feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]

    if _has(feats, ["A1_rt_mean","A1_acc_norm"]):
        feats["A1_speed_acc_tradeoff"] = _safe_div(feats["A1_rt_mean"], feats["A1_acc_norm"], eps)
    if _has(feats, ["A2_rt_mean","A2_acc_norm"]):
        feats["A2_speed_acc_tradeoff"] = _safe_div(feats["A2_rt_mean"], feats["A2_acc_norm"], eps)
    if _has(feats, ["A4_rt_mean","A4_acc_congruent"]):
        feats["A4_speed_acc_tradeoff"] = _safe_div(feats["A4_rt_mean"], feats["A4_acc_congruent"], eps)

    for k in ["A1","A2","A3","A4","A5"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if _has(feats, [m, s]):
            feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    if _has(feats, ["A1_rt_fast","A1_rt_slow"]):
        feats["A1_rt_speed_log_ratio"] = _log_ratio(feats["A1_rt_fast"], feats["A1_rt_slow"], eps)
    if _has(feats, ["A1_acc_fast","A1_acc_slow"]):
        feats["A1_acc_speed_log_ratio"] = _log_ratio(feats["A1_acc_fast"], feats["A1_acc_slow"], eps)

    if _has(feats, ["A2_rt_fast_c1","A2_rt_slow_c1"]):
        feats["A2_rt_speed_log_ratio_c1"] = _log_ratio(feats["A2_rt_fast_c1"], feats["A2_rt_slow_c1"], eps)
    if _has(feats, ["A2_acc_fast","A2_acc_slow"]):
        feats["A2_acc_speed_log_ratio"] = _log_ratio(feats["A2_acc_fast"], feats["A2_acc_slow"], eps)

    if _has(feats, ["A3_rt_big","A3_rt_small"]):
        feats["A3_rt_size_log_ratio"] = _log_ratio(feats["A3_rt_big"], feats["A3_rt_small"], eps)
    if _has(feats, ["A3_valid_acc","A3_invalid_acc"]):
        feats["A3_acc_attention_log_ratio"] = _log_ratio(feats["A3_valid_acc"], feats["A3_invalid_acc"], eps)

    if _has(feats, ["A4_rt_incongruent","A4_rt_congruent"]):
        feats["A4_stroop_rt_log_ratio"] = _log_ratio(feats["A4_rt_incongruent"], feats["A4_rt_congruent"], eps)
    if _has(feats, ["A4_acc_congruent","A4_acc_incongruent"]):
        feats["A4_stroop_acc_log_ratio"] = _log_ratio(feats["A4_acc_congruent"], feats["A4_acc_incongruent"], eps)

    if _has(feats, ["A5_acc_nonchange","A5_acc_pos_change"]):
        feats["A5_acc_pos_log_ratio"] = _log_ratio(feats["A5_acc_nonchange"], feats["A5_acc_pos_change"], eps)
    if _has(feats, ["A5_acc_nonchange","A5_acc_color_change"]):
        feats["A5_acc_color_log_ratio"] = _log_ratio(feats["A5_acc_nonchange"], feats["A5_acc_color_change"], eps)

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats

def add_features_B(df: pd.DataFrame) -> pd.DataFrame:
    feats = df.copy()
    eps = 1e-6
    if _has(feats, ["Year","Month"]):
        feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]

    if _has(feats, ["B1_rt_mean","B1_task1_acc"]):
        feats["B1_speed_acc_tradeoff"] = _safe_div(feats["B1_rt_mean"], feats["B1_task1_acc"], eps)
    if _has(feats, ["B2_rt_mean","B2_task1_acc"]):
        feats["B2_speed_acc_tradeoff"] = _safe_div(feats["B2_rt_mean"], feats["B2_task1_acc"], eps)
    if _has(feats, ["B3_rt_mean","B3_acc_rate"]):
        feats["B3_speed_acc_tradeoff"] = _safe_div(feats["B3_rt_mean"], feats["B3_acc_rate"], eps)
    if _has(feats, ["B4_rt_mean","B4_congruent_acc"]):
        feats["B4_speed_acc_tradeoff"] = _safe_div(feats["B4_rt_mean"], feats["B4_congruent_acc"], eps)
    if _has(feats, ["B5_rt_mean","B5_acc_rate"]):
        feats["B5_speed_acc_tradeoff"] = _safe_div(feats["B5_rt_mean"], feats["B5_acc_rate"], eps)

    for k in ["B1","B2","B3","B4","B5"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if _has(feats, [m, s]):
            feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    if _has(feats, ["B1_change_acc","B1_nonchange_acc"]):
        feats["B1_acc_log_ratio"] = _log_ratio(feats["B1_nonchange_acc"], feats["B1_change_acc"], eps)
    if _has(feats, ["B2_change_acc","B2_nonchange_acc"]):
        feats["B2_acc_log_ratio"] = _log_ratio(feats["B2_nonchange_acc"], feats["B2_change_acc"], eps)

    if _has(feats, ["B4_congruent_acc","B4_incongruent_acc"]):
        feats["B4_flanker_acc_log_ratio"] = _log_ratio(feats["B4_congruent_acc"], feats["B4_incongruent_acc"], eps)

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats

# -----------------------------
# 5. 히스토리 + 정규화 + Delta(B-only)
# -----------------------------
def add_history_and_norm(all_df: pd.DataFrame, norm_stats: dict) -> pd.DataFrame:
    df = all_df.copy()
    df["base_index"] = np.arange(len(df))

    if "YearMonthIndex" not in df.columns and {"Year","Month"}.issubset(df.columns):
        df["YearMonthIndex"] = df["Year"] * 12 + df["Month"]

    sort_cols = ["PrimaryKey"]
    if "Year" in df.columns:
        sort_cols += ["Year","Month"]
    sort_cols.append("Test_id")

    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Test 구분 컬럼
    if "Test_x" in df.columns:
        test_col = "Test_x"
    elif "Test" in df.columns:
        test_col = "Test"
    else:
        raise KeyError("히스토리 생성을 위해 'Test' 혹은 'Test_x' 컬럼이 필요합니다.")

    grp = df.groupby("PrimaryKey", sort=False)
    df["pk_hist_total_count"] = grp.cumcount()

    df["_is_A"] = (df[test_col] == "A").astype(int)
    df["_is_B"] = (df[test_col] == "B").astype(int)
    df["pk_hist_A_count"] = grp["_is_A"].cumsum().shift(1).fillna(0).astype(int)
    df["pk_hist_B_count"] = grp["_is_B"].cumsum().shift(1).fillna(0).astype(int)

    if "YearMonthIndex" in df.columns:
        df["pk_hist_prev_ym"] = grp["YearMonthIndex"].shift(1)
        df["pk_hist_gap_from_prev"] = df["YearMonthIndex"] - df["pk_hist_prev_ym"]

    df.drop(columns=["_is_A","_is_B"], inplace=True)

    df = df.sort_values("base_index").reset_index(drop=True)
    df.drop(columns=["base_index"], inplace=True)

    age_mean = norm_stats.get("Age_num_mean", None)
    age_std  = norm_stats.get("Age_num_std", None)
    if age_mean is not None and age_std is not None and "Age_num" in df.columns:
        df["Age_num_z"] = (df["Age_num"] - age_mean) / (age_std + 1e-6)

    ym_mean = norm_stats.get("YearMonthIndex_mean", None)
    ym_std  = norm_stats.get("YearMonthIndex_std", None)
    if ym_mean is not None and ym_std is not None and "YearMonthIndex" in df.columns:
        df["YearMonthIndex_z"] = (df["YearMonthIndex"] - ym_mean) / (ym_std + 1e-6)

    return df

def _delta_colname(c: str) -> str:
    return f"delta_{c.replace('-', '_')}"

def add_delta_features_pk(df: pd.DataFrame, test_col_name: str) -> pd.DataFrame:
    df = df.copy()
    b_prefixes = ("B1_","B2_","B3_","B4_","B5_","B6_","B7_","B8_","B9-","B10-")
    candidates = [c for c in df.columns if (c.startswith(b_prefixes)) and (df[c].dtype != "O")]

    sort_cols_local = ["PrimaryKey"]
    if "Year" in df.columns:
        sort_cols_local += ["Year","Month"]
    sort_cols_local.append("Test_id")

    df = df.sort_values(sort_cols_local, kind="mergesort").reset_index(drop=True)
    g = df.groupby("PrimaryKey", sort=False)

    for c in candidates:
        prev = g[c].shift(1)
        delta = df[c] - prev
        use_mask = df[test_col_name] == "B"
        df[_delta_colname(c)] = np.where(use_mask, delta, np.nan)

    return df

# -----------------------------
# 6. Exam-order features + prior
# -----------------------------
def detect_test_col(df: pd.DataFrame) -> str:
    if "Test_x" in df.columns:
        return "Test_x"
    elif "Test" in df.columns:
        return "Test"
    else:
        raise KeyError("A/B 구분을 위한 'Test' 또는 'Test_x' 컬럼이 필요합니다.")

def _exam_order_bucket(t, o):
    # A: 1 / 2 / 3+
    # B: 1 / 2-3 / 4+
    if t == "A":
        if o == 1:  return "A_1"
        if o == 2:  return "A_2"
        return "A_3plus"
    else:
        if o == 1:         return "B_1"
        if o in (2, 3):    return "B_2to3"
        return "B_4plus"

def add_exam_order_features(df: pd.DataFrame, test_col: str) -> pd.DataFrame:
    df = df.copy()
    df["__base__"] = np.arange(len(df))

    sort_cols = ["PrimaryKey"]
    if "Year" in df.columns:
        sort_cols += ["Year","Month"]
    sort_cols.append("Test_id")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    df["exam_order_in_test"] = df.groupby(["PrimaryKey", test_col]).cumcount() + 1
    df["is_first_in_test"] = (df["exam_order_in_test"] == 1).astype("Int8")
    df["exam_order_bucket"] = np.vectorize(_exam_order_bucket)(
        df[test_col].astype(str).values,
        df["exam_order_in_test"].astype(int).values
    )

    df = df.sort_values("__base__").drop(columns="__base__").reset_index(drop=True)
    return df

def add_prior_risk_by_order(df: pd.DataFrame, test_col: str) -> pd.DataFrame:
    df = df.copy()
    prior_path = os.path.join(MODEL_SAVE_DIR, "order_prior.csv")
    if os.path.exists(prior_path):
        prior = pd.read_csv(prior_path)
        # 기대 컬럼: [test_col, "exam_order_bucket", "prior_risk_by_order"]
        if test_col not in prior.columns and "Test" in prior.columns:
            prior = prior.rename(columns={"Test":"Test_x"})
        df = df.merge(prior[[test_col, "exam_order_bucket", "prior_risk_by_order"]],
                      on=[test_col, "exam_order_bucket"], how="left")
    else:
        df["prior_risk_by_order"] = df["exam_order_bucket"].map(DEFAULT_ORDER_PRIOR)

    df["prior_risk_by_order"] = df["prior_risk_by_order"].fillna(DEFAULT_GLOBAL_PRIOR).astype(float)
    return df

# -----------------------------
# 7. CatBoost 입력 준비/예측
# -----------------------------
def prepare_cb_input(df: pd.DataFrame, feature_names) -> pd.DataFrame:
    X = df.copy()
    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        for c in missing:
            if c in CAT_FEATURES:
                X[c] = "nan"
            else:
                X[c] = np.nan
    X = X[feature_names].copy()
    for col in CAT_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna("nan").astype(str)
    return X

def prepare_cat_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("nan").astype(str)
    return df

def predict_group(test_group_df: pd.DataFrame,
                  group_label: str,
                  n_splits: int = N_SPLITS) -> np.ndarray:
    test_group_df = prepare_cat_features(test_group_df)
    fold_preds = []
    valid_fold_count = 0

    for fold in range(n_splits):
        if group_label == "A":
            model_path = os.path.join(MODEL_SAVE_DIR, f"catboost_A_fold{fold}.pkl")
            calib_path = os.path.join(MODEL_SAVE_DIR, f"calibrator_A_fold{fold}.pkl")
        else:
            model_path = os.path.join(MODEL_SAVE_DIR, f"catboost_B_fold{fold}.pkl")
            calib_path = os.path.join(MODEL_SAVE_DIR, f"calibrator_B_fold{fold}.pkl")

        if not (os.path.exists(model_path) and os.path.exists(calib_path)):
            print(f"[WARN] Fold {fold} {group_label}: 모델/보정기 파일 없음, 건너뜀.")
            continue

        model = joblib.load(model_path)
        calibrator = joblib.load(calib_path)
        if model is None or calibrator is None:
            print(f"[INFO] Fold {fold} {group_label}: model/calibrator가 None, 건너뜀.")
            continue

        feat_names = getattr(model, "feature_names_", None)
        if feat_names is None:
            X = test_group_df.select_dtypes(include=[np.number]).copy()
        else:
            X = prepare_cb_input(test_group_df, feat_names)

        print(f"[INFO] Fold {fold} {group_label}: 예측 (rows={len(X)}, feats={X.shape[1]})")
        proba_raw = model.predict_proba(X)[:, 1]
        proba_cal = calibrator.predict(proba_raw)

        fold_preds.append(proba_cal)
        valid_fold_count += 1

    if valid_fold_count == 0:
        print(f"[WARN] {group_label}: 유효 fold 없음 → 0.5로 채움.")
        return np.full(len(test_group_df), 0.5, dtype=float)

    return np.mean(np.vstack(fold_preds), axis=0)

# -----------------------------
# 8. 메인 파이프라인
# -----------------------------
def main():
    print("[INFO] v19 inference script 시작.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 정규화 통계
    norm_stats_path = os.path.join(MODEL_SAVE_DIR, "normalization_stats.pkl")
    if not os.path.exists(norm_stats_path):
        raise FileNotFoundError(
            f"정규화 통계 파일 없음: {norm_stats_path}\n"
            " → 1_Preprocess_delta_logratio.py 실행 후 normalization_stats.pkl 포함 필요."
        )
    norm_stats = joblib.load(norm_stats_path)
    print(f"[INFO] normalization_stats 로드: {norm_stats_path}")

    # PK Stats (B 전용)
    pk_stats_path = os.path.join(MODEL_SAVE_DIR, "pk_stats_final.csv")
    if not os.path.exists(pk_stats_path):
        raise FileNotFoundError(
            f"PK Stats 파일 없음: {pk_stats_path}\n"
            " → 2_Train_Models_v19 실행 후 pk_stats_final.csv 포함 필요."
        )
    pk_stats_final = pd.read_csv(pk_stats_path)
    print(f"[INFO] pk_stats_final 로드: {pk_stats_path}, shape={pk_stats_final.shape}")

    # Test 데이터
    test_meta_path = os.path.join(BASE_DIR, "test.csv")
    test_A_path = os.path.join(BASE_DIR, "test", "A.csv")
    test_B_path = os.path.join(BASE_DIR, "test", "B.csv")

    test_meta = pd.read_csv(test_meta_path)
    test_A_raw = pd.read_csv(test_A_path)
    test_B_raw = pd.read_csv(test_B_path)
    print("[INFO] Test 로드 완료.", "meta:", test_meta.shape, "A:", test_A_raw.shape, "B:", test_B_raw.shape)

    # 1차/2차 피처
    print("[INFO] 도메인 피처 생성 (A/B)...")
    test_A_features = preprocess_A(test_A_raw)
    test_B_features = preprocess_B(test_B_raw)

    print("[INFO] log_ratio 등 2차 피처 생성...")
    test_A_features = add_features_A(test_A_features)
    test_B_features = add_features_B(test_B_features)

    # 메타 병합
    print("[INFO] 메타 병합...")
    meta_A = test_meta[test_meta["Test"] == "A"].reset_index(drop=True)
    meta_B = test_meta[test_meta["Test"] == "B"].reset_index(drop=True)
    meta_A_features = meta_A.merge(test_A_features, on="Test_id", how="left")
    meta_B_features = meta_B.merge(test_B_features, on="Test_id", how="left")
    all_test_df = pd.concat([meta_A_features, meta_B_features], sort=False).reset_index(drop=True)

    # 히스토리 + 정규화
    print("[INFO] PK 히스토리 + 정규화 피처 생성...")
    all_test_df = add_history_and_norm(all_test_df, norm_stats)

    # A/B 구분 컬럼
    test_col = detect_test_col(all_test_df)

    # Exam-order 파생 + prior
    print("[INFO] Exam-order 파생 + prior 주입...")
    all_test_df = add_exam_order_features(all_test_df, test_col=test_col)
    all_test_df = add_prior_risk_by_order(all_test_df, test_col=test_col)

    # Delta(B-only)
    print("[INFO] Delta(B-only) 피처 생성...")
    all_test_df = add_delta_features_pk(all_test_df, test_col_name=test_col)

    # A/B 분리 + B에 PK Stats merge
    test_A_df = all_test_df[all_test_df[test_col] == "A"].copy()
    test_B_df = all_test_df[all_test_df[test_col] == "B"].copy()
    if len(test_B_df) > 0:
        test_B_df = test_B_df.merge(pk_stats_final, on="PrimaryKey", how="left")

    print("[INFO] 최종 행수 → A:", len(test_A_df), "B:", len(test_B_df))

    # 예측
    print("[INFO] 모델 예측 시작 (5-fold + Isotonic)...")
    preds_A = predict_group(test_A_df, group_label="A", n_splits=N_SPLITS) if len(test_A_df) > 0 else np.array([])
    preds_B = predict_group(test_B_df, group_label="B", n_splits=N_SPLITS) if len(test_B_df) > 0 else np.array([])

    # 제출 파일 생성
    print("[INFO] submission 생성...")
    pred_dict = {}
    if len(test_A_df) > 0:
        for tid, p in zip(test_A_df["Test_id"].values, preds_A):
            pred_dict[tid] = float(p)
    if len(test_B_df) > 0:
        for tid, p in zip(test_B_df["Test_id"].values, preds_B):
            pred_dict[tid] = float(p)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission = test_meta[["Test_id"]].copy()
    submission["Label"] = submission["Test_id"].map(pred_dict).astype(float)
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"[INFO] 저장 완료: {SUBMISSION_PATH}")
    print("[INFO] inference 종료.")

if __name__ == "__main__":
    main()
