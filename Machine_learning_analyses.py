# -*- coding: utf-8 -*-
# ==================== Configure input and output paths here ====================
PATH_FILE = "data/example_input.xlsx"
OUT_DIR = "results"

# Excel 首行为列名；默认 A 列为 Outcome，B 列为 RFI。
# 每行必须代表一个独立样本。
OUTER_SPLITS = 5
INNER_SPLITS = 5
RANDOM_STATE = 42
# AUC采用旧版DeLong 95% CI；其余分类指标仍采用固定OOF预测的近似bootstrap区间。
# 两者均不涵盖重新训练模型的变异。bootstrap默认2000次，可提高至5000。
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260929

import json
import platform
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def clean_data(raw):
    """只排除缺失项；非法非缺失标签/RFI直接报错，避免静默修改数据。"""
    if raw.shape[1] < 2:
        raise ValueError("Excel 至少需要两列：Outcome 和 RFI。")
    raw = raw.reset_index(drop=True)
    data = pd.DataFrame({'excel_row': np.arange(len(raw)) + 2})
    label_text = raw.iloc[:, 0].astype('string').str.strip()
    rfi_text = raw.iloc[:, 1].astype('string').str.strip()
    label_missing = raw.iloc[:, 0].isna() | label_text.eq('').fillna(False)
    rfi_missing = raw.iloc[:, 1].isna() | rfi_text.eq('').fillna(False)
    numeric_label = pd.to_numeric(label_text, errors='coerce')
    mapped_label = label_text.str.lower().map({'healthy': 0, 'malt+': 1})
    labels = numeric_label.fillna(mapped_label)
    bad_label = ~label_missing & ~labels.isin([0, 1])
    rfi = pd.to_numeric(rfi_text, errors='coerce')
    finite_rfi = np.isfinite(rfi.to_numpy(dtype=float, na_value=np.nan))
    bad_rfi = ~rfi_missing & ~finite_rfi
    if bad_label.any():
        raise ValueError(f"非法标签，Excel 行号 {data.loc[bad_label, 'excel_row'].tolist()}。"
                         "仅接受 0/1 或 Healthy/MALT+；不会将小数标签截断为整数。")
    if bad_rfi.any():
        raise ValueError(f"RFI 必须是有限数值，Excel 行号 {data.loc[bad_rfi, 'excel_row'].tolist()}。")
    data['label'] = labels
    data['RFI'] = rfi
    reasons = np.where(label_missing & rfi_missing, 'missing_label_and_RFI',
                       np.where(label_missing, 'missing_label', 'missing_RFI'))
    missing = label_missing | rfi_missing
    excluded = data.loc[missing, ['excel_row']].copy()
    excluded['reason'] = reasons[missing.to_numpy(dtype=bool)]
    data = data.loc[~missing].copy().reset_index(drop=True)
    data['label'] = data['label'].astype(int)
    data['RFI'] = data['RFI'].astype(float)
    data.insert(0, 'id', np.arange(1, len(data) + 1))
    return data, excluded


def make_splits(X, y, n_splits, seed):
    if n_splits < 2:
        raise ValueError("交叉验证至少需要两折。")
    y = np.asarray(y)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("分析数据必须同时包含 class 0 和 class 1。")
    counts = np.bincount(y, minlength=2)
    if min(counts) < n_splits:
        raise ValueError(f"{n_splits} 折验证要求每类至少 {n_splits} 个独立样本；当前为 {list(counts)}。")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(X, y))
    for tr, te in splits:
        if len(np.unique(y[tr])) != 2 or len(np.unique(y[te])) != 2:
            raise ValueError("某折缺少一个类别，请减少折数或检查数据。")
    return splits


def make_model():
    # 保留原脚本参数，确保仅修正阈值选择时外层预测可以直接比较。
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=3000, solver='lbfgs', C=1e6,
                                   class_weight='balanced', random_state=RANDOM_STATE))
    ])


def fit_predict(X_train, y_train, X_test):
    model = make_model()
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    # 为与原脚本严格比较，保留原有截断。极端概率可能因此出现并列。
    # 它不是ROC的必要步骤；如改动此项，须重算并更新全部结果。
    return np.clip(prob, 1e-8, 1 - 1e-8)


def select_threshold(X_train, y_train, seed):
    """仅接收外层训练数据，在其内部生成OOF分数，再选择Youden阈值。"""
    counts = np.bincount(y_train, minlength=2)
    inner_n = min(INNER_SPLITS, int(min(counts)))
    splits = make_splits(X_train, y_train, inner_n, seed)
    inner_prob = np.full(len(y_train), np.nan)
    inner_fold = np.zeros(len(y_train), dtype=int)
    for i, (tr, va) in enumerate(splits, 1):
        inner_prob[va] = fit_predict(X_train.iloc[tr], y_train[tr], X_train.iloc[va])
        inner_fold[va] = i
    if not np.isfinite(inner_prob).all():
        raise RuntimeError("内层 OOF 预测不完整。")
    fpr, tpr, thresholds = roc_curve(y_train, inner_prob, drop_intermediate=False)
    # 只考虑有限阈值；并列最优时预先规定取最接近0.5者，再取较大者。
    finite = np.flatnonzero(np.isfinite(thresholds))
    youden = tpr - fpr
    best = finite[np.isclose(youden[finite], np.max(youden[finite]), rtol=0, atol=1e-12)]
    chosen = min(best, key=lambda k: (abs(thresholds[k] - 0.5), -thresholds[k]))
    audit = pd.DataFrame({'inner_fold': inner_fold, 'label': y_train, 'inner_oof_prob': inner_prob})
    return float(thresholds[chosen]), audit, inner_n


def evaluate_fold(X, y, tr, te, fold):
    y = np.asarray(y)
    threshold, audit, inner_n = select_threshold(
        X.iloc[tr], y[tr], RANDOM_STATE + fold)
    # 阈值确定后，外层模型仍使用完整外层训练集重新拟合。
    prob = fit_predict(X.iloc[tr], y[tr], X.iloc[te])
    pred = (prob >= threshold).astype(int)
    return {'threshold': threshold, 'prob': prob, 'pred': pred,
            'auc': roc_auc_score(y[te], prob), 'inner_audit': audit, 'inner_n': inner_n}


def classification_metrics(y, pred):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return cm, {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
                'Sensitivity': tp / (tp + fn), 'Specificity': tn / (tn + fp),
                'Accuracy': (tp + tn) / len(y)}


def delong_auc_ci(y, prob, level=0.95):
    """与旧版fast_delong等价的单条ROC DeLong方差及正态近似95%区间。"""
    y = np.asarray(y)
    prob = np.asarray(prob, dtype=float)
    if y.ndim != 1 or prob.ndim != 1 or len(y) != len(prob):
        raise ValueError('DeLong需要一维等长标签和预测分数。')
    if set(np.unique(y)) != {0, 1} or not np.isfinite(prob).all():
        raise ValueError('DeLong需要0/1标签和有限预测分数。')
    positive, negative = prob[y == 1], prob[y == 0]
    if min(len(positive), len(negative)) < 2:
        raise ValueError('DeLong每类至少需要两个独立样本。')
    pairwise = (positive[:, None] > negative[None, :]).astype(float)
    pairwise += 0.5 * (positive[:, None] == negative[None, :])
    v01, v10 = pairwise.mean(axis=1), pairwise.mean(axis=0)
    auc_value = float(v01.mean())
    variance = np.var(v01, ddof=1) / len(positive) + np.var(v10, ddof=1) / len(negative)
    if not np.isfinite(variance) or variance < 0:
        raise ValueError('DeLong方差无效。')
    margin = norm.ppf(1 - (1 - level) / 2) * np.sqrt(variance)
    return auc_value, max(0.0, auc_value - margin), min(1.0, auc_value + margin)


def bootstrap_units(y):
    """按结局分层构造独立样本索引。"""
    y = np.asarray(y)
    if y.ndim != 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError('Bootstrap需要同时包含0和1标签的一维数组。')
    strata = [np.flatnonzero(y == 0), np.flatnonzero(y == 1)]
    if min(map(len, strata)) < 2:
        raise ValueError('Bootstrap每类至少需要两个独立样本；小样本区间仍可能不稳定。')
    return strata


def draw_bootstrap_indices(strata, rng):
    return np.concatenate([
        rng.choice(indices, size=len(indices), replace=True)
        for indices in strata
    ])


def bootstrap_oof_ci(y, prob, pred, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """固定OOF分数及分类结果的近似区间；不再次优化阈值、不改变点估计。

    本方法不涵盖重新拟合、重新分折及重新选择阈值的不确定性。
    分层抽样固定两类独立样本数；准确率区间条件于该病例/对照构成，
    不代表另一患病率人群的准确率不确定性。
    """
    y, prob, pred = np.asarray(y), np.asarray(prob, dtype=float), np.asarray(pred)
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 100:
        raise ValueError('N_BOOTSTRAP须为至少100的整数，正式分析建议2000或更多。')
    if prob.ndim != 1 or pred.ndim != 1 or len(y) != len(prob) or len(y) != len(pred):
        raise ValueError('标签、概率、分类结果必须是一维等长数组。')
    if not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
        raise ValueError('预测概率必须是[0,1]内的有限值。')
    if not np.isin(pred, [0, 1]).all():
        raise ValueError('分类结果必须是0或1。')
    strata = bootstrap_units(y)
    rng = np.random.default_rng(seed)
    names = ['AUC_OOF', 'Sensitivity', 'Specificity', 'Accuracy']

    def metrics_at(ix):
        yy, pp = y[ix], pred[ix]
        return [roc_auc_score(yy, prob[ix]), np.mean(pp[yy == 1] == 1),
                np.mean(pp[yy == 0] == 0), np.mean(pp == yy)]

    samples = np.array([metrics_at(draw_bootstrap_indices(strata, rng)) for _ in range(n_boot)])
    bounds = np.quantile(samples, [0.025, 0.975], axis=0)
    table = pd.DataFrame({'Metric': names, 'Estimate': metrics_at(np.arange(len(y))),
                          'CI95_Lower': bounds[0], 'CI95_Upper': bounds[1]})
    table['Method'] = 'Approximate stratified percentile bootstrap of fixed OOF predictions'
    table['Resampling_unit'] = 'independent_row'
    table['N_bootstrap'] = n_boot
    table['Seed'] = seed
    if np.any(bounds[0] == bounds[1]):
        warnings.warn('至少一个bootstrap区间退化为单点。完美分类或小样本可导致此现象；'
                      '不能据此认为真实性能没有不确定性。', RuntimeWarning)
    return table, pd.DataFrame(samples, columns=names)


def save_cm(cm, path, title):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    ax.set(title=title, xlabel='Predicted label', ylabel='Actual label')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_roc_data(y, prob, path):
    fpr, tpr, thresholds = roc_curve(y, prob)
    pd.DataFrame({'FPR_1-Specificity': fpr, 'TPR_Sensitivity': tpr,
                  'Thresholds': thresholds}).to_excel(path, index=False)
    return fpr, tpr


def main(path_file=PATH_FILE, out_dir=OUT_DIR, n_boot=N_BOOTSTRAP):
    raw = pd.read_excel(path_file)
    use, excluded = clean_data(raw)
    X, y = use[['RFI']], use['label'].to_numpy()
    splits = make_splits(X, y, OUTER_SPLITS, RANDOM_STATE)
    run_dir = Path(out_dir) / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    data_dir = run_dir / '00_data_processed'
    model_dir = run_dir / 'GLM'
    cm_dir = model_dir / 'Confusion_Matrices_Per_Fold'
    roc_dir = model_dir / 'ROC_Curve_Data'
    threshold_dir = model_dir / 'Threshold_Training_Audit'
    for directory in [data_dir, cm_dir, roc_dir, threshold_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    use.to_csv(data_dir / 'cleaned_data.csv', index=False, encoding='utf-8-sig')
    excluded.to_csv(data_dir / 'excluded_rows.csv', index=False, encoding='utf-8-sig')
    print(f"原始 {len(raw)} 行；排除缺失 {len(excluded)} 行；纳入 {len(use)} 行。")
    print(f"Class 0={np.sum(y == 0)}，Class 1={np.sum(y == 1)}")
    print('分析单位：每行一个独立样本；使用分层交叉验证。')
    oof = np.full(len(y), np.nan)
    predictions = np.full(len(y), -1, dtype=int)
    fold_number = np.zeros(len(y), dtype=int)
    applied_threshold = np.full(len(y), np.nan)
    rows = []
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    for fold, (tr, te) in enumerate(splits, 1):
        result = evaluate_fold(X, y, tr, te, fold)
        prob, pred, threshold = result['prob'], result['pred'], result['threshold']
        oof[te], predictions[te] = prob, pred
        fold_number[te], applied_threshold[te] = fold, threshold
        cm, metrics = classification_metrics(y[te], pred)
        rows.append({'Fold': fold, 'N_test': len(te), 'AUC': result['auc'],
                     'Threshold_from_inner_CV': threshold, 'Inner_folds': result['inner_n'], **metrics})
        fpr, tpr = save_roc_data(y[te], prob, roc_dir / f'GLM_Fold_{fold}_ROC_Data.xlsx')
        ax.plot(fpr, tpr, lw=1, alpha=0.4, label=f'Fold {fold} (AUC={result["auc"]:.3f})')
        save_cm(cm, cm_dir / f'Fold_{fold}_Confusion_Matrix.png',
                f'GLM Fold {fold}: held-out predictions\n'
                f'Sens={metrics["Sensitivity"]:.2f}, Spec={metrics["Specificity"]:.2f}')
        audit = result['inner_audit'].copy()
        audit.insert(0, 'excel_row', use.iloc[tr]['excel_row'].to_numpy())
        audit.insert(0, 'id', use.iloc[tr]['id'].to_numpy())
        audit.to_csv(threshold_dir / f'fold{fold}_training_inner_oof.csv', index=False, encoding='utf-8-sig')
        fold_df = use.iloc[te].copy()
        fold_df['prob'], fold_df['pred'] = prob, pred
        fold_df['threshold_from_training'] = threshold
        fold_df.to_csv(model_dir / f'fold{fold}_pred.csv', index=False, encoding='utf-8-sig')
        print(f"Fold {fold}: AUC={result['auc']:.4f}；训练集内阈值={threshold:.4f}")
    if not np.isfinite(oof).all() or np.any(predictions < 0):
        raise RuntimeError('外层 OOF 结果不完整。')
    details = pd.DataFrame(rows)
    details.to_excel(model_dir / 'GLM_5Fold_Detailed_Metrics.xlsx', index=False)
    fpr, tpr = save_roc_data(y, oof, roc_dir / 'GLM_Overall_OOF_ROC_Data.xlsx')
    auc_oof = roc_auc_score(y, oof)
    print(f'计算AUC的DeLong 95%区间及其余指标的{n_boot}次bootstrap近似95%区间...')
    ci_table, bootstrap_draws = bootstrap_oof_ci(y, oof, predictions, n_boot=n_boot)
    delong_auc, auc_lower, auc_upper = delong_auc_ci(y, oof)
    if not np.isclose(delong_auc, auc_oof, atol=1e-12):
        raise RuntimeError('DeLong AUC与OOF AUC不一致。')
    auc_row = ci_table['Metric'].eq('AUC_OOF')
    ci_table.loc[auc_row, ['CI95_Lower', 'CI95_Upper']] = [auc_lower, auc_upper]
    ci_table.loc[auc_row, 'Method'] = 'DeLong normal-approximation CI of pooled OOF AUC'
    ci_table.loc[auc_row, ['N_bootstrap', 'Seed']] = np.nan
    ci_table.to_csv(run_dir / 'OOF_Metrics_95CI.csv', index=False, encoding='utf-8-sig')
    ci_table.to_excel(run_dir / 'OOF_Metrics_95CI.xlsx', index=False)
    bootstrap_draws.to_csv(model_dir / 'bootstrap_metric_draws.csv', index=False)
    auc_ci = ci_table.set_index('Metric').loc['AUC_OOF']
    ax.plot(fpr, tpr, lw=2.5, color='#a65628',
            label=f'OOF AUC={auc_oof:.3f}\nDeLong 95% CI: {auc_ci.CI95_Lower:.3f}-{auc_ci.CI95_Upper:.3f}')
    ax.plot([0, 1], [0, 1], '--', lw=2, color='r', alpha=0.8)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02),
           xlabel='False Positive Rate (1 - Specificity)', ylabel='True Positive Rate (Sensitivity)',
           title=f'GLM ROC Curves ({OUTER_SPLITS}-Fold CV)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(model_dir / 'GLM_ROC_5folds.png', dpi=300)
    fig.savefig(model_dir / 'GLM_ROC_5folds.pdf')
    plt.close(fig)
    cm, metrics = classification_metrics(y, predictions)
    save_cm(cm, model_dir / 'GLM_OOF_Confusion_Matrix_Heatmap.png',
            'GLM pooled held-out confusion matrix\nFold-specific thresholds selected in training data')
    use['outer_fold'], use['prob'] = fold_number, oof
    use['threshold_from_training'], use['pred'] = applied_threshold, predictions
    use.to_csv(model_dir / 'oof_predictions.csv', index=False, encoding='utf-8-sig')
    # 论文主结果只保留一个总体AUC：合并所有外层held-out OOF预测后计算的AUC。
    # 各折AUC仍保存在GLM_5Fold_Detailed_Metrics.xlsx中，仅用于折级审计。
    summary = pd.DataFrame([{'Model': 'GLM', 'N': len(y),
                             'AUC_OOF': auc_oof, **metrics}])
    for row in ci_table.itertuples(index=False):
        suffix = 'CI95' if row.Metric == 'AUC_OOF' else 'Approx_CI95'
        summary[f'{row.Metric}_{suffix}_Lower'] = row.CI95_Lower
        summary[f'{row.Metric}_{suffix}_Upper'] = row.CI95_Upper
    summary.to_csv(run_dir / 'Model_Performance_Summary.csv', index=False, encoding='utf-8-sig')
    # Record only the file name so public example outputs do not expose local paths.
    metadata = {'input_file': Path(path_file).name,
                'outer_folds': OUTER_SPLITS, 'inner_folds_max': INNER_SPLITS,
                'seed': RANDOM_STATE, 'python': platform.python_version(),
                'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
                'matplotlib': matplotlib.__version__, 'model': str(make_model()),
                'n_raw': len(raw), 'n_included': len(use), 'n_excluded_missing': len(excluded),
                'probability_clip': [1e-8, 1-1e-8],
                'threshold_selection': 'Youden index on inner OOF predictions from outer training data',
                'p_value': 'Not calculated',
                'confidence_intervals': {
                    'level': 0.95, 'auc_method': 'DeLong normal approximation of pooled OOF AUC',
                    'other_metrics_method': 'Approximate stratified percentile bootstrap of FIXED OOF predictions',
                    'n_bootstrap_other_metrics': n_boot, 'bootstrap_seed': BOOTSTRAP_SEED,
                    'unit': 'independent_row',
                    'model_refitted': False, 'threshold_reselected': False,
                    'metric_weighting': 'sample weighted',
                    'limitation': 'DeLong on pooled OOF scores and fixed-prediction bootstrap do not fully account for overlapping training sets or retraining uncertainty'}}
    (run_dir / 'analysis_settings.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    (run_dir / '结果解释.txt').write_text(
        '这是内部交叉验证结果，不是外部验证。\n'
        '每个外层测试折使用仅由其训练数据内层OOF预测选出的阈值。\n'
        '总体混淆矩阵汇总各折的独立测试预测，不在全部OOF标签上重新选阈值。\n'
        '论文主AUC采用AUC_OOF：将5个外层测试折的held-out OOF预测按原样本顺序合并后计算ROC AUC。\n'
        '每折AUC仅保存在详细指标文件中用于审计，不再计算或报告折间Mean AUC。\n'
        '没有计算模型显著性P值，不应在论文中自行补写。\n'
        'AUC采用旧版DeLong方差的正态近似95%区间；灵敏度、特异度和准确率仍采用固定OOF预测的分层percentile bootstrap近似95%区间。\n'
        f'仅其余三个指标的bootstrap次数={n_boot}，随机种子={BOOTSTRAP_SEED}；取2.5%与97.5%分位数。bootstrap_metric_draws.csv中的AUC抽样列不用于报告的AUC区间。\n'
        '每次抽样保持标签、预测分数和预先产生的分类结果绑定，不重新选阈值。\n'
        '这些区间不充分涵盖训练集重叠、重新分折、模型再训练和阈值重选的不确定性，可能偏窄。\n'
        '不能将它称为完整建模流程或独立外部验证的95%置信区间。\n'
        '准确率及其区间条件于研究样本的病例/对照构成，不代表其他患病率人群。\n'
        '完美预测时区间可能退化为[1,1]，不意味着真实人群性能确定为100%。\n'
        'class_weight=balanced下的概率不直接代表实际人群患病风险。\n'
        '本脚本假定每一行代表一个独立样本。\n'
        '仅比较MALT+与Healthy不能代表对其他疾病的鉴别诊断能力。\n'
        '本脚本没有建立或验证一个供未来使用的单一最终阈值/部署模型。\n', encoding='utf-8')
    print(summary.to_string(index=False))
    print(f'完成，结果目录：{run_dir.resolve()}')
    return run_dir


if __name__ == '__main__':
    main()
