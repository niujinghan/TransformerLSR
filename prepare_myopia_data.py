"""
prepare_myopia_data.py

将近视预测数据集（train/val/test CSV）转换为 TransformerLSR 所需的 .pkl 格式，
并生成 dag_info.pkl（纵向变量因果 DAG 结构）。

数据集结构说明
--------------
- 样本单元：sample_id（眼睛级别，每个患者双眼各一个 sample_id）
- 纵向变量（9个，全部随时间变化）：age, AL, K1, K2, WTW, SPH, CYL, AX, SE
- 基线变量（2个，不随时间变化）：gender, eye
- 时间列：t_from_baseline（单位：月，距基线的时间）
- 生存时间：tte（从 landmark 到事件/截尾的剩余时间，单位：月）
- 事件标志：event（1=近视发生，0=截尾）
- 访问序号：visit_num（1-indexed）
- 总访问数：seq_len（该观察窗口内的总访问次数）

重要说明
--------
数据集已按 split 列预先划分为 train/val/test，不应随机重新划分！
代码通过读取各 CSV 文件并分别保存 pkl 来保留原始分割。

运行方式
--------
    cd model/TransformerLSR
    python prepare_myopia_data.py
"""

import pandas as pd
import numpy as np
import pickle
import os

# ============================================================
# 配置区域
# ============================================================

# 纵向变量（全部随时间变化的临床测量，d_long=9）
LONG_COLS = ['AL', 'K1', 'K2', 'WTW', 'SPH', 'CYL', 'AX', 'SE', 'age']

# 基线变量（不随时间变化，d_base=2）
BASE_COLS = ['gender', 'eye']

# 列名映射
ID_COL      = 'sample_id'
TIME_COL    = 'tte'             # 生存时间（从 landmark 到事件/截尾的剩余时间，单位：月）
EVENT_COL   = 'event'
OBSTIME_COL = 't_from_baseline' # 观察时间（从基线开始，单位：月）
SEQLEN_COL  = 'seq_len'        # 该观察窗口内总访问次数
VISIT_COL   = 'visit_num'      # 访问序号（原为 1-indexed，转为 0-indexed）

# 路径配置
DATA_DIR    = '../../dataset/dynamic'
OUTPUT_DIR  = './data'
DATASET_NAME = 'myopia'
SEED        = 1

# DAG 配置（纵向变量间因果关系）
# 当前假设无因果边（独立假设）
# 若有先验知识（如 age -> AL, age -> SE），可在此设置 adj_matrix
N_LONG = len(LONG_COLS)
adj_matrix = np.zeros((N_LONG, N_LONG))

# ============================================================
# 内部函数
# ============================================================

def process_split(csv_path, id_offset=0):
    """读取单个 CSV，做列名映射，返回处理后的 DataFrame 和 id 映射。"""
    df = pd.read_csv(csv_path)

    # 计算真正的总生存时间 (total_time = t_from_baseline的最大值 + tte)
    # 因为原表中的 tte 只是从最后一次访问算起的剩余时间
    max_t = df.groupby(ID_COL)[OBSTIME_COL].transform('max')
    df['total_time'] = max_t + df[TIME_COL]

    # 列名重映射
    rename_map = {
        ID_COL:      'id',
        'total_time':'time',
        EVENT_COL:   'event',
        OBSTIME_COL: 'obstime',
        SEQLEN_COL:  'num_visit',
        VISIT_COL:   'visit',
    }
    for i, col in enumerate(LONG_COLS):
        rename_map[col] = f'Y{i+1}'
    for i, col in enumerate(BASE_COLS):
        rename_map[col] = f'X{i+1}'
    df = df.rename(columns=rename_map)

    # visit 转为 0-indexed
    df['visit'] = df['visit'] - 1

    # 重新编号 id（全局连续整数，从 id_offset 开始）
    unique_ids = sorted(df['id'].unique())
    id_map = {old_id: new_id + id_offset for new_id, old_id in enumerate(unique_ids)}
    df['id'] = df['id'].map(id_map)
    next_offset = id_offset + len(unique_ids)

    # 保留必要列
    keep_cols = (
        ['id', 'time', 'event', 'obstime', 'num_visit', 'visit']
        + [f'Y{i+1}' for i in range(N_LONG)]
        + [f'X{i+1}' for i in range(len(BASE_COLS))]
    )
    df = df[keep_cols]
    return df, next_offset


def build_dag_info(adj_matrix):
    """构建 dag_info 字典。"""
    try:
        import networkx as nx
        G = nx.DiGraph(adj_matrix)
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("adj_matrix 存在环，请检查 DAG 配置！")
        topo_order = list(nx.topological_sort(G))
    except ImportError:
        print("[警告] 未找到 networkx，使用默认拓扑序 [0,1,...,N-1]")
        topo_order = list(range(N_LONG))
    return {
        "dag": adj_matrix,
        "order": np.array(topo_order)
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  TransformerLSR 数据预处理（近视预测数据集）")
    print("=" * 60)

    # ---- 1. 分别处理三个 split ----
    print(f"\n[1/4] 读取 CSV 文件（来自 {DATA_DIR}）...")
    train_df, offset = process_split(os.path.join(DATA_DIR, 'train.csv'), id_offset=0)
    val_df,   offset = process_split(os.path.join(DATA_DIR, 'val.csv'),   id_offset=offset)
    test_df,  offset = process_split(os.path.join(DATA_DIR, 'test.csv'),  id_offset=offset)

    print(f"  train: {train_df['id'].nunique()} 个样本，{len(train_df)} 行")
    print(f"  val:   {val_df['id'].nunique()} 个样本，{len(val_df)} 行")
    print(f"  test:  {test_df['id'].nunique()} 个样本，{len(test_df)} 行")

    # ---- 2. 合并为 all（TransformerLSR 内部会按 id 切分，需要区分 split）----
    # 在合并数据中保留 split 标签（供 main.py 按 split 划分使用）
    train_df['split'] = 'train'
    val_df['split']   = 'val'
    test_df['split']  = 'test'
    data_all = pd.concat([train_df, val_df, test_df], ignore_index=True)
    # split_info: {id -> 'train'/'val'/'test'}
    split_info = data_all.drop_duplicates('id')[['id', 'split']].set_index('id')['split'].to_dict()
    data_all_no_split = data_all.drop(columns=['split'])

    print(f"\n[2/4] 数据统计...")
    event_rate = data_all.drop_duplicates('id')['event'].mean()
    print(f"  总样本数: {data_all['id'].nunique()}")
    print(f"  事件率:   {event_rate:.2%}")
    print(f"  tte 范围: [{data_all.drop_duplicates('id')['time'].min():.2f}, {data_all.drop_duplicates('id')['time'].max():.2f}] 月")
    print(f"  纵向变量: {LONG_COLS} → Y1~Y{N_LONG}")
    print(f"  基线变量: {BASE_COLS} → X1, X2")

    print(f"\n[3/4] 保存数据集 pkl...")
    output_path = os.path.join(OUTPUT_DIR, f'{DATASET_NAME}_seed_{SEED}.pkl')
    data_all_no_split.to_pickle(output_path)
    print(f"  → {output_path}")

    # 保存 split_info 以供 main.py 使用（按 split 列划分，而非随机划分）
    split_path = os.path.join(OUTPUT_DIR, f'{DATASET_NAME}_split_info.pkl')
    with open(split_path, 'wb') as f:
        pickle.dump(split_info, f)
    print(f"  → {split_path}（split 划分信息）")

    print(f"\n[4/4] 生成 DAG 信息...")
    dag_info = build_dag_info(adj_matrix)
    dag_path = os.path.join(OUTPUT_DIR, f'{DATASET_NAME}_info.pkl')
    with open(dag_path, 'wb') as f:
        pickle.dump(dag_info, f)
    print(f"  拓扑序: {dag_info['order'].tolist()}")
    print(f"  → {dag_path}")

    print("\n预处理完成！")
    print(f"   python main.py --data {DATASET_NAME} --d_long {N_LONG} --epoch 50")
    print("=" * 60)


if __name__ == '__main__':
    main()
