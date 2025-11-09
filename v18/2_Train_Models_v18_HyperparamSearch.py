## ----------------------------------------------------------------
## [Jupyter용] 2_Train_Models_v18_HyperparamSearch.ipynb
##  - 목적: v18용 A/B CatBoost 하이퍼파라미터 Random Search
##  - v18 구조 유지:
##      * A: PK Stats 미사용 (Base + Hist + Norm)
##      * B: PK Stats 사용 (Base + PK + Hist + Norm)
##  - 평가 지표:
##      * Combined Score = 0.5*(1-AUC) + 0.25*Brier + 0.25*ECE  (작을수록 좋음)
##  - 출력:
##      * A-model best params / best score
##      * B-model best params / best score
##    -> 이 값을 2_Train_Models_v18.ipynb 의 BEST_A_PARAMS / BEST_B_PARAMS 에 넣어서 최종 학습하면 됨
## ----------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Big-Tech ML Engineer v18 Hyperparam Search: 시작.")

# ------------------------------------------------
# 0. 경로 및 데이터 로드
# ------------------------------------------------
BASE_DIR = "./data"
MODEL_SAVE_DIR = "./model"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

FEATURE_SAVE_PATH = os.path.join(BASE_DIR, "all_train_data.feather")

print("[INFO] all_train_data.feather 로드 중...")
try:
    all_train_df = pd.read_feather(FEATURE_SAVE_PATH)
except FileNotFoundError as e:
    print(f"에러: {e}")
    print("먼저 1_Preprocess.ipynb 를 실행해서 all_train_data.feather 를 만들어야 합니다.")
    raise

print(f"[INFO] all_train_df shape: {all_train_df.shape}")

# Label 결측 방어
all_train_df["Label"] = all_train_df["Label"].fillna(0).astype(int)

# ------------------------------------------------
# 1. ECE / Combined Score 유틸
# ------------------------------------------------
def expected_calibration_error(y_true, y_prob, n_bins=10):
    if len(y_true) == 0 or len(y_prob) == 0:
        return 0.0
    y_prob = np.nan_to_num(y_prob, nan=0.0)

    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_edges[0] = -0.001
    bin_edges[-1] = 1.001

    df["y_prob"] = np.clip(df["y_prob"], 0, 1)
    df["bin"] = pd.cut(df["y_prob"], bins=bin_edges, right=True)

    bin_stats = df.groupby("bin", observed=True).agg(
        bin_total=("y_prob", "count"),
        prob_true=("y_true", "mean"),
        prob_pred=("y_prob", "mean"),
    )
    non_empty = bin_stats[bin_stats["bin_total"] > 0]
    if len(non_empty) == 0:
        return 0.0

    weights = non_empty["bin_total"] / len(y_prob)
    prob_true = non_empty["prob_true"]
    prob_pred = non_empty["prob_pred"]
    ece = np.sum(weights * np.abs(prob_true - prob_pred))
    return ece


def combined_score(y_true, y_prob):
    """
    - AUC는 높을수록 좋음 → 1 - AUC
    - Brier / ECE는 낮을수록 좋음
    최종 Combined Score도 낮을수록 좋음.
    """
    if (
        len(y_true) == 0
        or len(y_prob) == 0
        or np.sum(y_true) == 0
        or np.sum(y_true) == len(y_true)
    ):
        print("  AUC: N/A (단일 클래스), Brier: N/A, ECE: N/A (No data)")
        return 1.0

    y_prob = np.nan_to_num(y_prob, nan=0.0)
    mean_auc = roc_auc_score(y_true, y_prob)
    mean_brier = mean_squared_error(y_true, y_prob)
    mean_ece = expected_calibration_error(y_true, y_prob)
    score = 0.5 * (1 - mean_auc) + 0.25 * mean_brier + 0.25 * mean_ece

    print(f"  AUC: {mean_auc:.4f}, Brier: {mean_brier:.4f}, ECE: {mean_ece:.4f}")
    print(f"  Combined Score: {score:.5f}")
    return score

# ------------------------------------------------
# 2. 공통 세팅: K-Fold / CatBoost 입력 구성
# ------------------------------------------------
print("\n[INFO] StratifiedKFold 5 splits 준비 중...")
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

train_indices_list = []
val_indices_list = []

for train_idx, val_idx in skf.split(all_train_df, all_train_df["Label"]):
    train_indices_list.append(train_idx)
    val_indices_list.append(val_idx)

print("[INFO] StratifiedKFold 5 splits 준비 완료.")

CAT_FEATURES = ["Age", "PrimaryKey"]
DROP_COLS_TRAIN = [
    "Test_id",
    "Test_x",
    "Test_y",
    "Label",
    "TestDate",
    "Year",
    "Month",
    "base_index",
]

def _build_cb_matrices(X_train, X_val):
    """
    공통: numeric + categorical 분리 및 cat feature 인덱스 계산
    (A: PK Stats 없이, B: PK Stats merge 후 호출)
    """
    numeric_cols = list(set(X_train.columns) - set(CAT_FEATURES) - set(DROP_COLS_TRAIN))
    # train/val 공통으로 있는 numeric만 사용 (안전)
    common_numeric_cols = list(set(numeric_cols) & set(X_val.columns))

    cb_X_train = X_train[common_numeric_cols + CAT_FEATURES].copy()
    cb_X_val = X_val[common_numeric_cols + CAT_FEATURES].copy()

    for col in CAT_FEATURES:
        if col in cb_X_train.columns:
            cb_X_train[col] = cb_X_train[col].fillna("nan").astype(str)
            cb_X_val[col] = cb_X_val[col].fillna("nan").astype(str)

    cat_indices = [
        cb_X_train.columns.get_loc(c)
        for c in CAT_FEATURES
        if c in cb_X_train.columns
    ]
    return cb_X_train, cb_X_val, cat_indices

# ------------------------------------------------
# 3. PK Stats 생성 함수 (B 모델용, fold마다)
# ------------------------------------------------
def build_pk_stats_for_fold(train_df_fold):
    """
    - v18 학습 코드와 동일한 로직으로 PK Stats 생성
    - 이 fold의 train 데이터에서만 집계하여 leakage 방지
    """
    agg_funcs = {
        "Age_num": ["mean", "min", "max"],
        "YearMonthIndex": ["mean", "std", "min", "max"],
        "A1_rt_mean": ["mean", "std"],
        "A4_acc_congruent": ["mean", "std"],
        "A4_acc_incongruent": ["mean", "std"],
        "A4_stroop_rt_cost": ["mean", "std"],
        "RiskScore": ["mean", "std", "max"],
        "B1_change_acc": ["mean", "std"],
        "B1_nonchange_acc": ["mean", "std"],
        "B3_rt_mean": ["mean", "std"],
        "B4_flanker_acc_cost": ["mean", "std"],
        "B4_rt_mean": ["mean", "std"],
        "RiskScore_B": ["mean", "std", "max"],
        "Test_id": ["count"],
    }
    # 실제 존재하는 컬럼만 사용
    valid_agg_funcs = {
        col: funcs
        for col, funcs in agg_funcs.items()
        if col in train_df_fold.columns
    }

    pk_stats_fold = train_df_fold.groupby("PrimaryKey").agg(valid_agg_funcs)
    pk_stats_fold.columns = [
        "_".join(col).strip() for col in pk_stats_fold.columns.values
    ]
    if "Test_id_count" in pk_stats_fold.columns:
        pk_stats_fold.rename(columns={"Test_id_count": "pk_test_total_count"}, inplace=True)

    # PK별 A/B 시험 횟수 (train fold 내부에서만)
    if "Test_x" in train_df_fold.columns:
        test_col = "Test_x"
    elif "Test" in train_df_fold.columns:
        test_col = "Test"
    else:
        raise KeyError("PK Stats 생성을 위해 'Test' 혹은 'Test_x' 컬럼이 필요합니다.")

    pk_test_type_count_fold = (
        train_df_fold.groupby("PrimaryKey")[test_col].value_counts().unstack(fill_value=0)
    )
    if "A" not in pk_test_type_count_fold.columns:
        pk_test_type_count_fold["A"] = 0
    if "B" not in pk_test_type_count_fold.columns:
        pk_test_type_count_fold["B"] = 0
    pk_test_type_count_fold = pk_test_type_count_fold[["A", "B"]]
    pk_test_type_count_fold.columns = ["pk_test_A_count", "pk_test_B_count"]

    pk_stats_fold = pk_stats_fold.join(pk_test_type_count_fold, how="left").reset_index()
    return pk_stats_fold

# ------------------------------------------------
# 4. A 모델용 CV 함수
# ------------------------------------------------
def run_cv_A(cat_params, iterations=2000):
    """
    - A: PK Stats 미사용, Base + Hist + Norm 피처만 사용
    - 각 fold에서 Test == 'A' 인 데이터만 사용
    - 5-fold OOF prediction 기반으로 Combined Score 계산
    """
    oof_pred_list = []
    oof_y_list = []

    print(f"[A] CatBoost params: {cat_params}")

    for fold in range(N_SPLITS):
        train_idx = train_indices_list[fold]
        val_idx = val_indices_list[fold]

        train_df_fold = all_train_df.iloc[train_idx].copy()
        val_df_fold = all_train_df.iloc[val_idx].copy()

        # Test 구분 컬럼
        if "Test_x" in train_df_fold.columns:
            test_col = "Test_x"
        elif "Test" in train_df_fold.columns:
            test_col = "Test"
        else:
            raise KeyError("A 모델 학습을 위해 'Test' 혹은 'Test_x' 컬럼이 필요합니다.")

        X_train_A = train_df_fold[train_df_fold[test_col] == "A"].copy()
        X_val_A = val_df_fold[val_df_fold[test_col] == "A"].copy()

        y_train_A = X_train_A["Label"].values
        y_val_A = X_val_A["Label"].values

        if len(X_train_A) == 0 or len(X_val_A) == 0:
            print(f"  [A][Fold {fold+1}] A 데이터가 부족하여 이 Fold는 건너뜁니다.")
            continue
        if len(np.unique(y_train_A)) < 2 or len(np.unique(y_val_A)) < 2:
            print(f"  [A][Fold {fold+1}] 단일 클래스 Fold라서 스킵합니다.")
            continue

        cb_X_train, cb_X_val, cat_indices = _build_cb_matrices(X_train_A, X_val_A)

        model = cb.CatBoostClassifier(
            iterations=iterations,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            thread_count=-1,
            early_stopping_rounds=100,
            verbose=False,
            **cat_params,
        )
        model.fit(
            cb_X_train,
            y_train_A,
            eval_set=[(cb_X_val, y_val_A)],
            cat_features=cat_indices,
        )

        pred_val = model.predict_proba(cb_X_val)[:, 1]
        oof_pred_list.append(pred_val)
        oof_y_list.append(y_val_A)

    if not oof_pred_list:
        print("[A] 유효한 Fold가 없어 score를 계산할 수 없습니다.")
        return 1.0

    y_true_all = np.concatenate(oof_y_list)
    y_pred_all = np.concatenate(oof_pred_list)

    print("[A] 전체 Fold OOF 기준 Combined Score:")
    cv_score = combined_score(y_true_all, y_pred_all)
    return cv_score

# ------------------------------------------------
# 5. B 모델용 CV 함수 (PK Stats 포함)
# ------------------------------------------------
def run_cv_B(cat_params, iterations=2000):
    """
    - B: PK Stats 사용 + Base + Hist + Norm
    - 각 fold마다:
        1) train_df_fold 에서 PK Stats 생성
        2) Test == 'B' 만 뽑아서 PK Stats merge
        3) OOF prediction 기반 Combined Score
    """
    oof_pred_list = []
    oof_y_list = []

    print(f"[B] CatBoost params: {cat_params}")

    for fold in range(N_SPLITS):
        train_idx = train_indices_list[fold]
        val_idx = val_indices_list[fold]

        train_df_fold = all_train_df.iloc[train_idx].copy()
        val_df_fold = all_train_df.iloc[val_idx].copy()

        # Test 구분 컬럼
        if "Test_x" in train_df_fold.columns:
            test_col = "Test_x"
        elif "Test" in train_df_fold.columns:
            test_col = "Test"
        else:
            raise KeyError("B 모델 학습을 위해 'Test' 혹은 'Test_x' 컬럼이 필요합니다.")

        # --- PK Stats (train fold에서만) ---
        pk_stats_fold = build_pk_stats_for_fold(train_df_fold)

        # --- B 데이터만 ---
        X_train_B = train_df_fold[train_df_fold[test_col] == "B"].copy()
        X_val_B = val_df_fold[val_df_fold[test_col] == "B"].copy()

        y_train_B = X_train_B["Label"].values
        y_val_B = X_val_B["Label"].values

        if len(X_train_B) == 0 or len(X_val_B) == 0:
            print(f"  [B][Fold {fold+1}] B 데이터가 부족하여 이 Fold는 건너뜁니다.")
            continue
        if len(np.unique(y_train_B)) < 2 or len(np.unique(y_val_B)) < 2:
            print(f"  [B][Fold {fold+1}] 단일 클래스 Fold라서 스킵합니다.")
            continue

        # B에는 PK Stats merge
        X_train_B = X_train_B.merge(pk_stats_fold, on="PrimaryKey", how="left")
        X_val_B = X_val_B.merge(pk_stats_fold, on="PrimaryKey", how="left")

        cb_X_train, cb_X_val, cat_indices = _build_cb_matrices(X_train_B, X_val_B)

        model = cb.CatBoostClassifier(
            iterations=iterations,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            thread_count=-1,
            early_stopping_rounds=100,
            verbose=False,
            **cat_params,
        )
        model.fit(
            cb_X_train,
            y_train_B,
            eval_set=[(cb_X_val, y_val_B)],
            cat_features=cat_indices,
        )

        pred_val = model.predict_proba(cb_X_val)[:, 1]
        oof_pred_list.append(pred_val)
        oof_y_list.append(y_val_B)

    if not oof_pred_list:
        print("[B] 유효한 Fold가 없어 score를 계산할 수 없습니다.")
        return 1.0

    y_true_all = np.concatenate(oof_y_list)
    y_pred_all = np.concatenate(oof_pred_list)

    print("[B] 전체 Fold OOF 기준 Combined Score:")
    cv_score = combined_score(y_true_all, y_pred_all)
    return cv_score

# ------------------------------------------------
# 6. Random Search 설정 (A/B 각각)
#    - 예전 base 튜닝 결과를 중심으로 한 탐색 범위
# ------------------------------------------------
import itertools

# A 모델: v15에서 best
# {'depth': 6, 'learning_rate': 0.07, 'l2_leaf_reg': 3,
#  'random_strength': 0.5, 'bagging_temperature': 0.5, 'border_count': 128}
param_space_A = {
    "depth": [4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.07],
    "l2_leaf_reg": [3, 5, 10],
    "random_strength": [0.5, 1.0, 2.0],
    "bagging_temperature": [0.3, 0.5, 1.0],
    "border_count": [128, 254],
}

# B 모델: v15에서 best
# {'depth': 3, 'learning_rate': 0.03, 'l2_leaf_reg': 5,
#  'random_strength': 3.0, 'bagging_temperature': 1.0, 'border_count': 128}
param_space_B = {
    "depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05, 0.07],
    "l2_leaf_reg": [3, 5, 10],
    "random_strength": [1.0, 2.0, 3.0],
    "bagging_temperature": [0.3, 0.5, 1.0],
    "border_count": [128, 254],
}

def make_param_grid(space):
    keys = list(space.keys())
    values = [space[k] for k in keys]
    combos = list(itertools.product(*values))
    grid = []
    for combo in combos:
        params = {k: v for k, v in zip(keys, combo)}
        grid.append(params)
    return grid

grid_A = make_param_grid(param_space_A)
grid_B = make_param_grid(param_space_B)

rng = np.random.RandomState(2025)
rng.shuffle(grid_A)
rng.shuffle(grid_B)

N_TRIALS_A = 20   # 필요시 늘리거나 줄이면 됨
N_TRIALS_B = 20

# ------------------------------------------------
# 7. Random Search: A 모델
# ------------------------------------------------
best_score_A = np.inf
best_params_A = None

print("\n================ A 모델 Random Search 시작 ================")
print(f"[INFO] 후보 조합 수: {len(grid_A)}, 실제 실행 trial: {N_TRIALS_A}")

for i, params in enumerate(grid_A[:N_TRIALS_A], start=1):
    print(f"\n[A] Trial {i}/{N_TRIALS_A} params={params}")
    cv_score = run_cv_A(params, iterations=2000)  # 튜닝 시에는 2000 정도로 시간 절약
    print(f"[A] -> CV Combined Score = {cv_score:.5f}")

    if cv_score < best_score_A:
        best_score_A = cv_score
        best_params_A = params
        print(f"[A] ★ Update best! score={best_score_A:.5f}, params={best_params_A}")

print("\n[A] Random Search 완료.")
print(f"[A] Best Score = {best_score_A}")
print(f"[A] Best Params = {best_params_A}")

# ------------------------------------------------
# 8. Random Search: B 모델
# ------------------------------------------------
best_score_B = np.inf
best_params_B = None

print("\n================ B 모델 Random Search 시작 ================")
print(f"[INFO] 후보 조합 수: {len(grid_B)}, 실제 실행 trial: {N_TRIALS_B}")

for i, params in enumerate(grid_B[:N_TRIALS_B], start=1):
    print(f"\n[B] Trial {i}/{N_TRIALS_B} params={params}")
    cv_score = run_cv_B(params, iterations=2000)
    print(f"[B] -> CV Combined Score = {cv_score:.5f}")

    if cv_score < best_score_B:
        best_score_B = cv_score
        best_params_B = params
        print(f"[B] ★ Update best! score={best_score_B:.5f}, params={best_params_B}")

print("\n[B] Random Search 완료.")
print(f"[B] Best Score = {best_score_B}")
print(f"[B] Best Params = {best_params_B}")

# ------------------------------------------------
# 9. 최종 요약 출력
# ------------------------------------------------
print("\n================ 최종 결과 요약 ================")
print(f"A-model best params: {best_params_A}   CV score: {best_score_A}")
print(f"B-model best params: {best_params_B}   CV score: {best_score_B}")
print("→ 2_Train_Models_v18.ipynb 의 BEST_A_PARAMS / BEST_B_PARAMS 에 이 값들을 넣어주면 됩니다.")
