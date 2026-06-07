"""
eval.py — TransformerLSR 评估脚本（近视预测数据集适配版）

评价指标与 TrajSurv 完全一致（固定评估时间点为 6, 9, 12 个月）：
  1. C-index (concordance_index_censored)
  2. Time-dependent AUC @ 6m, 9m, 12m (cumulative_dynamic_auc)
  3. Brier Score @ 6m, 9m, 12m (brier_score)
  4. Integrated Brier Score (integrated_brier_score)
  5. RMSE of longitudinal predictions (Y1~Y9)

运行方式：
    python eval.py --data myopia --d_long 9 [其他参数同 main.py]
"""

import torch
import argparse
import logging
import time
import os
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2_dist

from TransformerLSR import TransformerLSR
from functions import get_tensors

# 与 TrajSurv 完全相同的指标库
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    brier_score,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

pd.options.mode.chained_assignment = None


# ============================================================
# 评估时间点（与 TrajSurv 完全一致）
# ============================================================
EVAL_TIMES = [6.0, 9.0, 12.0]   # 单位：月


def make_et_array(e_np, t_np):
    """将 numpy 数组转换为 sksurv 所需的结构化数组。"""
    return np.array(
        [(bool(e), float(t)) for e, t in zip(e_np, t_np)],
        dtype=[('e', bool), ('t', float)]
    )


def compute_surv_metrics(et_train, et_eval, pred_surv, times, logger, prefix="Test"):
    """
    计算并打印所有生存分析指标，与 TrajSurv.evaluate_dataset_metrics 对齐。

    参数
    ----
    et_train  : sksurv 结构化数组，训练集 (event, time)
    et_eval   : sksurv 结构化数组，评估集 (event, time)
    pred_surv : ndarray [n_samples, n_times]，预测生存概率
    times     : List[float]，评估时间点
    """
    risk_scores = 1.0 - pred_surv   # 风险得分 = 1 - 生存概率

    logger.info("\n" + "=" * 55)
    logger.info(f"=== {prefix} 集 生存分析专业指标评估 ===")
    logger.info("=" * 55)

    # 1. C-index
    try:
        c_index = concordance_index_censored(
            et_eval['e'], et_eval['t'], risk_scores[:, -1]
        )[0]
        logger.info(f"[1] C-index:                    {c_index:.4f}")
    except Exception as ex:
        c_index = 0.0
        logger.info(f"[1] C-index: Error - {ex}")

    # 过滤超出范围的时间点
    t_min, t_max = et_eval['t'].min(), et_eval['t'].max()
    valid_times   = [t for t in times if t_min < t < t_max]
    valid_indices = [times.index(t) for t in valid_times]

    auc_list = [0.0] * len(times)
    bs_list  = [0.0] * len(times)
    ibs      = 0.0

    if valid_times:
        valid_risk   = risk_scores[:, valid_indices]
        valid_surv   = pred_surv[:, valid_indices]

        # 2. Time-dependent AUC
        try:
            auc, _ = cumulative_dynamic_auc(et_train, et_eval, valid_risk, valid_times)
            for t, a, idx in zip(valid_times, auc, valid_indices):
                auc_list[idx] = float(a)
                logger.info(f"[2] AUC @ {t:>5.1f}m:                {a:.4f}")
        except Exception as ex:
            logger.info(f"[2] AUC: Error - {ex}")

        # 3. Brier Score
        try:
            _, bs = brier_score(et_train, et_eval, valid_surv, valid_times)
            for t, b, idx in zip(valid_times, bs, valid_indices):
                bs_list[idx] = float(b)
                logger.info(f"[3] Brier Score @ {t:>5.1f}m:        {b:.4f}")
        except Exception as ex:
            logger.info(f"[3] Brier Score: Error - {ex}")

        # 4. Integrated Brier Score
        try:
            ibs = float(integrated_brier_score(et_train, et_eval, valid_surv, valid_times))
            logger.info(f"[4] Integrated Brier Score (IBS): {ibs:.4f}")
        except Exception as ex:
            logger.info(f"[4] IBS: Error - {ex}")
    else:
        logger.info("[WARNING] 无有效评估时间点（所有时间点超出测试集范围）")

    logger.info("=" * 55 + "\n")
    return c_index, auc_list, bs_list, ibs


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",         default=1,      type=int)
    parser.add_argument("--batch_size",   default=16,     type=int)
    parser.add_argument("--num_enc_layer",default=4,      type=int)
    parser.add_argument("--num_dec_layer",default=4,      type=int)
    parser.add_argument("--num_head",     default=4,      type=int)
    parser.add_argument("--model_size",   default=32,     type=int)
    parser.add_argument("--d_long",       default=9,      type=int,   # 9个纵向变量
                        help="纵向变量数: AL,K1,K2,WTW,SPH,CYL,AX,SE,age")
    parser.add_argument('--suffix',       type=str,       default='eval')
    parser.add_argument('--model',        type=str,       default='LSR')
    parser.add_argument('--data',         type=str,       default='myopia')
    parser.add_argument("--local",        action="store_true")
    parser.add_argument("--inten_weight", default=0.01,   type=float)
    parser.add_argument("--surv_weight",  default=0.1,    type=float)
    parser.add_argument("--lr",           default=0.0003, type=float)
    # Y_missing 参数保留（兼容命令行，但真实数据不使用）
    parser.add_argument("--Y1_missing",   default=0,      type=float)
    parser.add_argument("--Y2_missing",   default=0,      type=float)
    parser.add_argument("--Y3_missing",   default=0,      type=float)
    args = parser.parse_args()

    # ---- Logger ----
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    work_dir = os.path.join('./work_dir', time.strftime("%Y-%m-%d", time.localtime()))
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    if not args.local:
        time_prefix = time.strftime("%H:%M:%S", time.localtime())
        log_path = (
            f"{work_dir}/{time_prefix}_{args.data}_{args.model}"
            f"_head_{args.num_head}_enc_{args.num_enc_layer}_dec_{args.num_dec_layer}"
            f"_size_{args.model_size}_{args.suffix}-log.txt"
        )
        fHandler = logging.FileHandler(log_path, mode='w')
        fHandler.setLevel(logging.DEBUG)
        fHandler.setFormatter(formatter)
        logger.addHandler(fHandler)

    logger.info(args)

    # ---- 变量配置 ----
    # 纵向变量：Y1=AL, Y2=K1, ..., Y9=age（顺序与 prepare_myopia_data.py LONG_COLS 对应）
    Y_str_list    = [f"Y{i+1}" for i in range(args.d_long)]
    # 基线变量：X1=gender, X2=eye
    BASE_str_list = ["X1", "X2"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed   = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # ---- 加载 DAG info ----
    with open(f'data/{args.data}_info.pkl', 'rb') as f:
        dag_info = pickle.load(f)

    # ---- 加载数据集 ----
    data_all = pd.read_pickle(f'data/{args.data}_seed_{seed}.pkl')
    I = data_all["id"].nunique()

    logger.info('=' * 55)
    logger.info(f'Starting evaluation for dataset: {args.data}')
    logger.info(f'{args.num_head} heads, {args.num_enc_layer} enc layers, '
                f'{args.num_dec_layer} dec layers, {args.model_size} model dimension')
    logger.info(f'Total trajectories: {I}')
    logger.info('=' * 55)

    # ---- 按预定义 split 划分（不随机重分！）----
    with open(f'data/{args.data}_split_info.pkl', 'rb') as f:
        split_info = pickle.load(f)   # {sample_id -> 'train'/'val'/'test'}

    data = data_all[data_all.obstime <= data_all.time]

    train_id = [sid for sid, sp in split_info.items() if sp == 'train']
    vali_id  = [sid for sid, sp in split_info.items() if sp == 'val']
    test_id  = [sid for sid, sp in split_info.items() if sp == 'test']

    train_data = data[data["id"].isin(train_id)]
    vali_data  = data[data["id"].isin(vali_id)]
    test_data  = data[data["id"].isin(test_id)]

    # MinMax 归一化（仅纵向变量，用 train_data 拟合）
    minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data.loc[:, Y_str_list] = minmax_scaler.fit_transform(train_data.loc[:, Y_str_list])
    vali_data.loc[:,  Y_str_list] = minmax_scaler.transform(vali_data.loc[:,  Y_str_list])
    test_data.loc[:,  Y_str_list] = minmax_scaler.transform(test_data.loc[:,  Y_str_list])

    # ---- 用于生存分析评估的 sksurv 结构化数组 ----
    # 从 train_data 中每个 id 取 event/time（取 visit==0 对应行）
    train_et_df = train_data[train_data["visit"] == 0][["event", "time"]]
    test_et_df  = test_data[test_data["visit"] == 0][["event", "time"]]

    et_train = make_et_array(train_et_df["event"].values.astype(bool),
                             train_et_df["time"].values.astype(float))
    et_test  = make_et_array(test_et_df["event"].values.astype(bool),
                             test_et_df["time"].values.astype(float))

    # ---- 地标时间与预测时间点 ----
    # 评估时间点与 TrajSurv 一致：6, 9, 12 个月
    # 地标时间：使用训练集生存时间的 10th 百分位
    LT = float(np.quantile(train_data["time"].drop_duplicates().values, 0.1))
    pred_times = [LT + t for t in EVAL_TIMES]
    logger.info(f"Landmark time (LT): {LT:.2f} 月")
    logger.info(f"Evaluation times:   {pred_times} 月")

    # ---- 加载模型 ----
    model_save_path = (
        f"./models/{args.data}_seed{seed}_{args.model}"
        f"_head_{args.num_head}_enc_layer_{args.num_enc_layer}"
        f"_dec_layer_{args.num_dec_layer}_size_{args.model_size}"
        f"_visit_weight_{args.inten_weight}_surv_weight_{args.surv_weight}"
        f"_lr_{args.lr}"
        f"Y1miss_{args.Y1_missing}Y2miss_{args.Y2_missing}Y3miss_{args.Y3_missing}.pt"
    )
    model = TransformerLSR(
        d_long=args.d_long, d_base=len(BASE_str_list),
        dag_info=dag_info, d_model=args.model_size,
        nhead=args.num_head, num_encoder_layers=args.num_enc_layer,
        num_decoder_layers=args.num_dec_layer, device=device
    )
    model.to(device=device)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    logger.info(f"Model loaded from: {model_save_path}")

    batch_size = args.batch_size
    d_long = args.d_long

    # ================================================================
    # Part 1: 纵向变量预测 RMSE（在 test_data 全集上）
    # ================================================================
    logger.info("\n--- Part 1: 纵向变量预测 RMSE ---")
    rmse_sum   = {f"Y{i+1}": 0.0 for i in range(d_long)}
    rmse_tokens = {f"Y{i+1}": 0   for i in range(d_long)}

    test_id_arr = list(test_id)
    for start in range(0, len(test_id_arr), batch_size):
        indices   = test_id_arr[start:start + batch_size]
        batch_data = test_data[test_data["id"].isin(indices)]
        batch      = get_tensors(batch_data.copy(), long=Y_str_list,
                                  base=BASE_str_list, device=device)
        with torch.no_grad():
            long_preds = model.predict_next_long_treat(batch)

        mask      = batch["mask"][:, 1:]
        long_true = batch["long"][:, 1:]
        bs_, l_, d_ = long_true.shape[0], long_true.shape[1], long_true.shape[2]
        nan_mask     = torch.isnan(long_true)
        combined_m   = (~nan_mask) & mask.unsqueeze(-1).repeat(1, 1, d_)

        y_hat    = long_preds.detach().cpu().numpy()
        y_target = long_true.cpu().numpy()
        cm_np    = combined_m.cpu().numpy()
        nm_np    = nan_mask.cpu().numpy()

        y_hat[nm_np]    = 0.0
        y_target[nm_np] = 0.0

        # 逆归一化
        y_hat_inv    = minmax_scaler.inverse_transform(
            y_hat.reshape(bs_ * l_, d_)).reshape(bs_, l_, d_)
        y_target_inv = minmax_scaler.inverse_transform(
            y_target.reshape(bs_ * l_, d_)).reshape(bs_, l_, d_)

        for i in range(d_long):
            mask_i   = cm_np[:, :, i].reshape(-1) > 0
            y_hat_i  = y_hat_inv[:, :, i].reshape(-1)[mask_i]
            y_tgt_i  = y_target_inv[:, :, i].reshape(-1)[mask_i]
            rmse_sum[f"Y{i+1}"]    += float(np.sum((y_hat_i - y_tgt_i) ** 2))
            rmse_tokens[f"Y{i+1}"] += int(mask_i.sum())

    for i in range(d_long):
        key = f"Y{i+1}"
        if rmse_tokens[key] > 0:
            rmse = np.sqrt(rmse_sum[key] / rmse_tokens[key])
            logger.info(f"  RMSE {key}: {rmse:.4f}  (n={rmse_tokens[key]})")

    # ================================================================
    # Part 2: 生存分析指标（landmark 模式，与 TrajSurv 对齐）
    # ================================================================
    logger.info("\n--- Part 2: 生存分析指标（Landmark 预测）---")

    # 只保留 time > LT 且 obstime <= LT 的 test 样本
    tmp_data  = test_data.loc[(test_data["time"] > LT) & (test_data["obstime"] <= LT), :]
    surv_id   = tmp_data["id"].unique()

    logger.info(f"  Landmark 样本数: {len(surv_id)}")

    if len(surv_id) == 0:
        logger.info("  [警告] landmark 样本数为 0，跳过生存分析评估")
        return

    # tmp_batch 用于获取真实 e, t（用于指标计算）
    tmp_batch = get_tensors(tmp_data.copy(), long=Y_str_list,
                             base=BASE_str_list)
    et_test_lm = make_et_array(
        tmp_batch["e"].numpy().astype(bool),
        tmp_batch["t"].numpy().astype(float)
    )

    # 预测生存函数
    total_pred = []
    surv_at_ti_list = []
    all_xlen_list = []

    surv_id_list = list(surv_id)
    for start in range(0, len(surv_id_list), batch_size):
        indices   = surv_id_list[start:start + batch_size]
        batch_data = tmp_data[tmp_data["id"].isin(indices)]
        batch      = get_tensors(batch_data.copy(), long=Y_str_list,
                                  base=BASE_str_list, device=device, eval_mode=True)

        _bs = batch["base"].shape[0]
        surv_pred = torch.zeros(_bs, 0, 1, device=device)
        start_time = LT

        for pt in pred_times:
            with torch.no_grad():
                surv_out = model.predict_surv_marginal(batch, end_time=pt,
                                                        start_time=start_time)
            surv_pred  = torch.cat((surv_pred, surv_out.unsqueeze(-1)), dim=1)
            start_time = pt

        surv_pred_np = surv_pred.squeeze(-1).cpu().numpy().reshape(_bs, -1)
        # cumsum of intensity → survival function S(t) = exp(-Λ(t))
        surv_pred_np = np.exp(-surv_pred_np.cumsum(axis=1))
        total_pred.append(surv_pred_np)

        # Get survival prob at exact t_true for D-calibration
        with torch.no_grad():
            surv_out_true = model.predict_surv_marginal(batch, end_time=batch["t"].unsqueeze(1), start_time=LT)
        surv_at_true = np.exp(-surv_out_true.squeeze(-1).cpu().numpy().reshape(_bs))
        surv_at_ti_list.extend(surv_at_true)

        # Calculate seq_len for stratified evaluation
        all_xlen_list.extend(torch.sum(batch["mask"], dim=-1).cpu().numpy())

    total_pred = np.concatenate(total_pred, axis=0)  # [n_test_lm, n_pred_times]
    surv_at_ti = np.array(surv_at_ti_list)
    all_xlen = np.array(all_xlen_list)

    # 计算指标
    c_index, auc_list, bs_list, ibs = compute_surv_metrics(
        et_train, et_test_lm, total_pred, pred_times, logger, prefix="Test (landmark)"
    )

    # Calibration Plot
    def kaplan_meier_at_t(events, times, t_star):
        order = np.argsort(times)
        sorted_times = times[order]
        sorted_events = events[order]
        surv = 1.0
        i = 0
        n = len(times)
        while i < n and sorted_times[i] <= t_star:
            t_j = sorted_times[i]
            n_j = n - i
            d_j = 0
            c_j = 0
            while i < n and sorted_times[i] == t_j:
                if sorted_events[i]:
                    d_j += 1
                else:
                    c_j += 1
                i += 1
            if n_j > 0 and d_j > 0:
                surv *= (1.0 - d_j / n_j)
        return surv

    def calibration_plot(pred_surv, et_test, eval_times, output_dir, dataset_name="Test", n_groups=10):
        events = et_test['e']
        times = et_test['t']
        n_times = len(eval_times)
        fig, axes = plt.subplots(1, n_times, figsize=(5 * n_times, 5))
        if n_times == 1:
            axes = [axes]
        for idx, (t_star, ax) in enumerate(zip(eval_times, axes)):
            pred_at_t = pred_surv[:, idx]
            order = np.argsort(pred_at_t)
            groups = np.array_split(order, n_groups)
            predicted, observed = [], []
            for group in groups:
                if len(group) < 2:
                    continue
                predicted.append(np.mean(pred_at_t[group]))
                observed.append(kaplan_meier_at_t(events[group], times[group], t_star))
            ax.scatter(predicted, observed, s=60, zorder=5, edgecolors='k', linewidths=0.5)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect')
            ax.set_xlabel('Predicted S(t)', fontsize=12)
            ax.set_ylabel('Observed KM S(t)', fontsize=12)
            ax.set_title(f'{dataset_name} @ {t_star:.0f}m', fontsize=13)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_aspect('equal')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, f'calibration_{dataset_name.replace(" ", "_").lower()}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    cal_path = calibration_plot(total_pred, et_test_lm, pred_times, "results", "Test")

    # D-calibration
    def compute_d_calibration(surv_at_ti, events, n_bins=10):
        events = events.astype(bool)
        cdf_values = np.clip(1.0 - surv_at_ti[events], 0, 1)
        if len(cdf_values) < n_bins:
            return None, None, None
        observed, _ = np.histogram(cdf_values, bins=np.linspace(0, 1, n_bins + 1))
        expected = len(cdf_values) / n_bins
        chi2_stat = float(np.sum((observed - expected) ** 2 / expected))
        p_value = float(1.0 - chi2_dist.cdf(chi2_stat, df=n_bins - 1))
        return chi2_stat, p_value, observed

    chi2_val, p_val, bin_counts = compute_d_calibration(surv_at_ti, tmp_batch["e"].numpy())

    # Stratified evaluation by seq_len
    stratified_results = []
    groups = [
        ('seq_len=1', all_xlen == 1),
        ('seq_len=2~3', (all_xlen >= 2) & (all_xlen <= 3)),
        ('seq_len≥4', all_xlen >= 4),
    ]
    y_test_e = tmp_batch["e"].numpy().astype(bool)
    y_test_t = tmp_batch["t"].numpy().astype(float)
    risk_scores = 1.0 - total_pred[:, -1]
    
    for name, grp_mask in groups:
        n = int(grp_mask.sum())
        n_events = int(y_test_e[grp_mask].sum())
        if n < 10 or n_events < 2:
            stratified_results.append(f"- **{name}** (n={n}, events={n_events}): 样本不足，跳过")
            continue
        try:
            ci = concordance_index_censored(y_test_e[grp_mask], y_test_t[grp_mask], risk_scores[grp_mask])[0]
            stratified_results.append(f"- **{name}** (n={n}, events={n_events}): C-index = `{ci:.4f}`")
        except Exception as ex:
            stratified_results.append(f"- **{name}** (n={n}): Error: {ex}")

    # ================================================================
    # Part 3: 保存结果
    # ================================================================
    results = {
        "c_index":    c_index,
        "auc":        {f"{t}m": a for t, a in zip(pred_times, auc_list)},
        "brier_score":{f"{t}m": b for t, b in zip(pred_times, bs_list)},
        "ibs":        ibs,
        "rmse":       {f"Y{i+1}": (np.sqrt(rmse_sum[f"Y{i+1}"] / rmse_tokens[f"Y{i+1}"])
                                    if rmse_tokens[f"Y{i+1}"] > 0 else float('nan'))
                       for i in range(d_long)},
    }

    result_path = (
        f"./results/{args.data}_seed{seed}_{args.model}"
        f"_head_{args.num_head}_enc_{args.num_enc_layer}"
        f"_dec_{args.num_dec_layer}_size_{args.model_size}.pkl"
    )
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"\n结果已保存至: {result_path}")

    # 打印汇总
    logger.info("\n========== 评估汇总 ==========")
    logger.info(f"C-index:   {c_index:.4f}")
    logger.info(f"Mean AUC:  {np.mean(auc_list):.4f}")
    logger.info(f"Mean BS:   {np.mean(bs_list):.4f}")
    logger.info(f"IBS:       {ibs:.4f}")
    
    logger.info(f"\n### 校准曲线")
    logger.info(f"- 校准曲线已保存至 `{cal_path}`")
    
    logger.info(f"\n### D-calibration 检验")
    if p_val is not None:
        status = "✅ 校准良好" if p_val > 0.05 else "⚠️ 校准不良"
        logger.info(f"- χ² = `{chi2_val:.2f}`, **p = `{p_val:.4f}`** ({status})")
        logger.info(f"- 各桶计数: `{bin_counts.tolist()}`")
    else:
        logger.info("- 已发病样本不足，无法计算")
        
    logger.info("\n### 按就诊次数 (seq_len) 分层 C-index")
    for res in stratified_results:
        logger.info(res)
        
    logger.info("=" * 30)

if __name__ == '__main__':
    main()