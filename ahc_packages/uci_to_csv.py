import pandas as pd
from sklearn.datasets import fetch_openml


def save_ucidata_to_csv(dataset_id, csv_path):
    # 从 UCI OpenML 数据库下载数据集
    dataset = fetch_openml(data_id=dataset_id, as_frame=True)

    # 获取数据和目标
    features = dataset.data
    targets = dataset.target

    # 将目标列添加到特征数据中
    features['Class'] = targets

    # 保存到 CSV 文件
    features.to_csv(csv_path, index=False)


# # 示例调用
# dataset_id = 545  # 数据集 ID（例如，你提到的 rice+cammeo+and+osmancik 数据集）
# csv_path = 'rice_cammeo_osmancik.csv'
# save_ucidata_to_csv(dataset_id, csv_path)