## ----------------------------------------------------------------
## [Jupyter용] 2_Train_Models_v20_5seeds.py
##  - v20 베이스 + 5-seeds 앙상블 학습
##  - 전략 2: PK 히스토리 피처 (Fold 내에서 생성, Leakage 방지)
##  - 전략 3: Age/YearMonthIndex 정규화 점수
##  - *_cost → *_log_ratio (로그 비율)
##  - Delta 피처: PK 기준 직전 관측과의 차이 (B만 사용)
##  - AgeGroup / Age_num_z2 / Age_num_z3 / YM_is_early 사용
##  - v20 규칙:
##      * A 모델: AgeGroup, YM_is_early 사용
##      * B 모델: AgeGroup, YM_is_early 제거 (나머지 피처 동일)
##  - 5-seeds:
##      * SEEDS = [42, 43, 44, 45, 46]
##      * 각 fold마다 seed별로 모델 저장
## ----------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Big-Tech ML Engineer (v20-5seeds): K-Fold + Leakage Fix + Domain Features 시작.")

## 0. Hyperparameter 설정 (v18 튜닝 결과)
BEST_A_PARAMS = {
    "depth": 6,
    "learning_rate": 0.07,
    "l2_leaf_reg": 10,
    "random_strength": 0.5,
    "bagging_temperature": 0.5,
    "border_count": 128,
}

BEST_B_PARAMS = {
    "depth": 5,
    "learning_rate": 0.03,
    "l2_leaf_reg": 3,
    "random_strength": 3.0,
    "bagging_temperature": 0.5,
    "border_count": 128,
}

# 5-seeds 설정
SEEDS = [42, 43, 44, 45, 46]

BASE_DIR = "./data"
MODEL_SAVE_DIR = "./model"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

FEATURE_SAVE_PATH = os.path.join(BASE_DIR, "all_train_data.feather")

try:
    print("전처리된 메인 피처 로드 중...")
    all_train_df = pd.read_feather(FEATURE_SAVE_PATH)
except FileNotFoundError as e:
    print(f"경고: {e}")
    print("먼저 1_Preprocess_v20.py를 실행하여 all_train_data.feather를 생성해야 합니다.")
    raise
print("데이터 로드 완료.")

def expected_calibration_error(y_true, y_prob, n_bins=10):
    if len(y_true) == 0 or len(y_prob) == 0:
        return 0.0
    y_prob = np.nan_to_num(y_prob, nan=0.0)
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_edges[0]  = -0.001
    bin_edges[-1] = 1.001
    df['y_prob'] = np.clip(df['y_prob'], 0, 1)
    df['bin'] = pd.cut(df['y_prob'], bins=bin_edges, right=True)
    bin_stats = df.groupby('bin', observed=True).agg(
        bin_total=('y_prob', 'count'),
        prob_true=('y_true', 'mean'),
        prob_pred=('y_prob', 'mean')
    )
    non_empty = bin_stats[bin_stats['bin_total'] > 0]
    if len(non_empty) == 0:
        return 0.0
    weights   = non_empty['bin_total'] / len(y_prob)
    prob_true = non_empty['prob_true']
    prob_pred = non_empty['prob_pred']
    ece = np.sum(weights * np.abs(prob_true - prob_pred))
    return ece

def combined_score(y_true, y_prob):
    if len(y_true) == 0 or len(y_prob) == 0 or np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
        print("  AUC: N/A (단일 클래스), Brier: N/A, ECE: N/A (No data)")
        return 1.0
    y_prob = np.nan_to_num(y_prob, nan=0.0)
    mean_auc   = roc_auc_score(y_true, y_prob)
    mean_brier = mean_squared_error(y_true, y_prob)
    mean_ece   = expected_calibration_error(y_true, y_prob)
    score = 0.5 * (1 - mean_auc) + 0.25 * mean_brier + 0.25 * mean_ece
    print(f"  AUC: {mean_auc:.4f}, Brier: {mean_brier:.4f}, ECE: {mean_ece:.4f}")
    print(f"  Combined Score: {score:.5f}")
    return score

print("\n[INFO] K-Fold 교차 검증 분리 시작...")
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

train_indices_list = []
val_indices_list   = []

all_train_df['Label'] = all_train_df['Label'].fillna(0)

for train_idx, val_idx in skf.split(all_train_df, all_train_df['Label']):
    train_indices_list.append(train_idx)
    val_indices_list.append(val_idx)

print(f"{N_SPLITS}-Fold 분리 완료.")

CAT_FEATURES_A    = ['Age', 'AgeGroup', 'PrimaryKey']
CAT_FEATURES_B    = ['Age', 'PrimaryKey']
DROP_COLS_TRAIN   = ['Test_id', 'Test_x', 'Test_y', 'Label', 'TestDate', 'Year', 'Month', 'base_index']

def _is_delta_column(col: str) -> bool:
    return col.startswith("delta_")

def _build_cb_matrices(X_train, X_val, cat_features):
    numeric_cols = list(set(X_train.columns) - set(cat_features) - set(DROP_COLS_TRAIN))
    common_numeric_cols = list(set(numeric_cols) & set(X_val.columns))
    cb_X_train = X_train[common_numeric_cols + cat_features].copy()
    cb_X_val   = X_val[common_numeric_cols + cat_features].copy()
    for col in cat_features:
        if col in cb_X_train.columns:
            train_col = cb_X_train[col].astype("object")
            val_col   = cb_X_val[col].astype("object")
            cb_X_train[col] = train_col.fillna("nan").astype(str)
            cb_X_val[col]   = val_col.fillna("nan").astype(str)
    cat_indices = [cb_X_train.columns.get_loc(c) for c in cat_features if c in cb_X_train.columns]
    return cb_X_train, cb_X_val, cat_indices

def train_model_A(X_train, y_train, X_val, y_val, seed, group_label="A"):
    drop_delta = [c for c in X_train.columns if _is_delta_column(c)]
    if drop_delta:
        X_train = X_train.drop(columns=drop_delta, errors="ignore")
        X_val   = X_val.drop(columns=drop_delta, errors="ignore")
    cb_X_train, cb_X_val, cat_features_indices = _build_cb_matrices(X_train, X_val, CAT_FEATURES_A)
    print(f"\n[{group_label}] CatBoost (Base+Hist+Norm+log_ratio+AgeGroup+YM_early, no PK, no Delta) "
          f"학습 시작... (seed={seed}, 피처 {len(cb_X_train.columns)}개)")
    cat_base_model = cb.CatBoostClassifier(
        iterations=3000,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=seed,
        thread_count=-1,
        early_stopping_rounds=100,
        verbose=1000,
        **BEST_A_PARAMS,
    )
    cat_base_model.fit(
        cb_X_train, y_train,
        eval_set=[(cb_X_val, y_val)],
        cat_features=cat_features_indices
    )
    print(f"\n[{group_label}] 단독 확률 보정 (Isotonic) 시작... (seed={seed})")
    pred_uncal = cat_base_model.predict_proba(cb_X_val)[:, 1]
    print(f"[{group_label}] 비보정 단독 점수 (seed={seed}):")
    _ = combined_score(y_val, pred_uncal)
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    calibrator.fit(pred_uncal, y_val)
    pred_cal = calibrator.predict(pred_uncal)
    print(f"[{group_label}] ★보정된 단독★ 최종 점수 (seed={seed}):")
    _ = combined_score(y_val, pred_cal)
    return cat_base_model, calibrator

def train_model_B(X_train, y_train, X_val, y_val, pk_stats_fold, seed, group_label="B"):
    X_train = X_train.merge(pk_stats_fold, on='PrimaryKey', how='left')
    X_val   = X_val.merge(pk_stats_fold,   on='PrimaryKey', how='left')
    drop_for_B = ['AgeGroup', 'YM_is_early']
    X_train = X_train.drop(columns=[c for c in drop_for_B if c in X_train.columns], errors='ignore')
    X_val   = X_val.drop(columns=[c for c in drop_for_B if c in X_val.columns], errors='ignore')
    cb_X_train, cb_X_val, cat_features_indices = _build_cb_matrices(X_train, X_val, CAT_FEATURES_B)
    n_delta_train = sum(_is_delta_column(c) for c in cb_X_train.columns)
    print(f"\n[{group_label}] Delta 사용 열 수: {n_delta_train}")
    print(f"[{group_label}] CatBoost (Base+PK+Hist+Norm+Delta+log_ratio, no AgeGroup/YM_early) "
          f"학습 시작... (seed={seed}, 피처 {len(cb_X_train.columns)}개)")
    cat_base_model = cb.CatBoostClassifier(
        iterations=3000,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=seed,
        thread_count=-1,
        early_stopping_rounds=100,
        verbose=1000,
        **BEST_B_PARAMS,
    )
    cat_base_model.fit(
        cb_X_train, y_train,
        eval_set=[(cb_X_val, y_val)],
        cat_features=cat_features_indices
    )
    print(f"\n[{group_label}] 단독 확률 보정 (Isotonic) 시작... (seed={seed})")
    pred_uncal = cat_base_model.predict_proba(cb_X_val)[:, 1]
    print(f"[{group_label}] 비보정 단독 점수 (seed={seed}):")
    _ = combined_score(y_val, pred_uncal)
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    calibrator.fit(pred_uncal, y_val)
    pred_cal = calibrator.predict(pred_uncal)
    print(f"[{group_label}] ★보정된 단독★ 최종 점수 (seed={seed}):")
    _ = combined_score(y_val, pred_cal)
    return cat_base_model, calibrator

all_pk_stats_folds = []

for fold in range(N_SPLITS):
    print(f"\n=== Fold {fold+1}/{N_SPLITS} 학습 시작 ===")
    train_idx = train_indices_list[fold]
    val_idx   = val_indices_list[fold]
    train_df_fold = all_train_df.iloc[train_idx].copy()
    val_df_fold   = all_train_df.iloc[val_idx].copy()
    print(f"\n[Fold {fold+1}] K-Fold Target Encoding (PK Stats) 생성...")

    agg_funcs = {
        'Age_num': ['mean', 'min', 'max'],
        'YearMonthIndex': ['mean', 'std', 'min', 'max'],
        'A1_rt_mean': ['mean', 'std'],
        'A4_acc_congruent': ['mean', 'std'],
        'A4_acc_incongruent': ['mean', 'std'],
        'A4_stroop_rt_log_ratio': ['mean', 'std'],
        'RiskScore': ['mean', 'std', 'max'],
        'B1_change_acc': ['mean', 'std'],
        'B1_nonchange_acc': ['mean', 'std'],
        'B3_rt_mean': ['mean', 'std'],
        'B4_flanker_acc_log_ratio': ['mean', 'std'],
        'B4_rt_mean': ['mean', 'std'],
        'RiskScore_B': ['mean', 'std', 'max'],
        'Test_id': ['count']
    }
    valid_agg_funcs = {col: funcs for col, funcs in agg_funcs.items() if col in train_df_fold.columns}
    pk_stats_fold = train_df_fold.groupby('PrimaryKey').agg(valid_agg_funcs)
    pk_stats_fold.columns = ['_'.join(col).strip() for col in pk_stats_fold.columns.values]
    pk_stats_fold.rename(columns={'Test_id_count': 'pk_test_total_count'}, inplace=True)

    if "Test_x" in train_df_fold.columns:
        test_col = "Test_x"
    elif "Test" in train_df_fold.columns:
        test_col = "Test"
    else:
        raise KeyError("PK Stats 생성을 위해 'Test' 혹은 'Test_x' 컬럼이 필요합니다.")

    pk_test_type_count_fold = train_df_fold.groupby('PrimaryKey')[test_col].value_counts().unstack(fill_value=0)
    if 'A' not in pk_test_type_count_fold.columns:
        pk_test_type_count_fold['A'] = 0
    if 'B' not in pk_test_type_count_fold.columns:
        pk_test_type_count_fold['B'] = 0
    pk_test_type_count_fold = pk_test_type_count_fold[['A', 'B']]
    pk_test_type_count_fold.columns = ['pk_test_A_count', 'pk_test_B_count']

    pk_stats_fold = pk_stats_fold.join(pk_test_type_count_fold, how='left').reset_index()
    all_pk_stats_folds.append(pk_stats_fold)

    X_train_A = train_df_fold[train_df_fold[test_col] == 'A'].copy()
    y_train_A = X_train_A['Label'].values
    X_val_A   = val_df_fold[val_df_fold[test_col] == 'A'].copy()
    y_val_A   = X_val_A['Label'].values

    X_train_B = train_df_fold[train_df_fold[test_col] == 'B'].copy()
    y_train_B = X_train_B['Label'].values
    X_val_B   = val_df_fold[val_df_fold[test_col] == 'B'].copy()
    y_val_B   = X_val_B['Label'].values

    # ---- 5 seeds 루프 ----
    for seed in SEEDS:
        print(f"\n--- 모델 A (seed={seed}) (PK Stats 미사용, Delta 제외, log_ratio + AgeGroup + YM_early 포함) 학습 ---")
        cat_A, calib_A = train_model_A(X_train_A, y_train_A, X_val_A, y_val_A,
                                       seed=seed, group_label=f"A_fold{fold}_s{seed}")
        joblib.dump(cat_A,   os.path.join(MODEL_SAVE_DIR, f"catboost_A_seed{seed}_fold{fold}.pkl"))
        joblib.dump(calib_A, os.path.join(MODEL_SAVE_DIR, f"calibrator_A_seed{seed}_fold{fold}.pkl"))

        print(f"\n--- 모델 B (seed={seed}) (PK Stats + Delta + log_ratio, AgeGroup/YM_early 제거) 학습 ---")
        if len(X_train_B) > 0 and len(X_val_B) > 0 and len(np.unique(y_train_B)) > 1:
            cat_B, calib_B = train_model_B(X_train_B, y_train_B, X_val_B, y_val_B,
                                           pk_stats_fold, seed=seed,
                                           group_label=f"B_fold{fold}_s{seed}")
            joblib.dump(cat_B,   os.path.join(MODEL_SAVE_DIR, f"catboost_B_seed{seed}_fold{fold}.pkl"))
            joblib.dump(calib_B, os.path.join(MODEL_SAVE_DIR, f"calibrator_B_seed{seed}_fold{fold}.pkl"))
        else:
            print(f"[Fold {fold+1}] (seed={seed}) B모델 학습/검증 데이터가 부족하여 이 Fold는 건너뜁니다.")
            joblib.dump(None, os.path.join(MODEL_SAVE_DIR, f"catboost_B_seed{seed}_fold{fold}.pkl"))
            joblib.dump(None, os.path.join(MODEL_SAVE_DIR, f"calibrator_B_seed{seed}_fold{fold}.pkl"))

print("\n[INFO] K-Fold PK Stats 병합...")
if all_pk_stats_folds:
    all_pk_stats_df = pd.concat(all_pk_stats_folds, ignore_index=True)
    final_pk_stats  = all_pk_stats_df.groupby('PrimaryKey').mean().reset_index()
    final_pk_stats_path = os.path.join(MODEL_SAVE_DIR, "pk_stats_final.csv")
    final_pk_stats.to_csv(final_pk_stats_path, index=False)
else:
    print("경고: 유효한 PK Stats가 생성되지 않았습니다. 빈 파일을 생성합니다.")
    final_pk_stats_path = os.path.join(MODEL_SAVE_DIR, "pk_stats_final.csv")
    pd.DataFrame().to_csv(final_pk_stats_path, index=False)

print("\n[INFO] '최종 보정' CatBoost v20-5seeds 모델 (A/B * 5seeds * 5fold) 및 최종 PK 통계 피처 저장 완료:")
print(f"  - PK Stats: {final_pk_stats_path}")
print("Big-Tech ML Engineer (v20-5seeds): 미션 완료.")
