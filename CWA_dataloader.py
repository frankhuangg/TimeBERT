import os
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class CWADataset(Dataset):
    def __init__(self, hdf5_path, metadata, transform=None):
        """
        初始化 CWA 數據集（metadata 為已過濾的 DataFrame）
        """
        self.hdf5_path = hdf5_path
        self.h5_file = None  # Lazy open
        self.transform = transform

        if isinstance(metadata, str):
            metadata = pd.read_csv(metadata)
        elif not isinstance(metadata, pd.DataFrame):
            raise TypeError("metadata 必須是 CSV 路徑或 pandas DataFrame")

        self.metadata = metadata.reset_index(drop=True)
        self.trace_names = self.metadata["trace_name"].values
        self.p_arrival_samples = self.metadata["trace_p_arrival_sample"].values
        self.pga_values = self.metadata["trace_pga_cmps2"].values

    def __len__(self):
        return len(self.trace_names)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.hdf5_path, "r")

        trace_name = self.trace_names[idx]
        p_arrival = self.p_arrival_samples[idx]
        pga_value = self.pga_values[idx]

        if pga_value < 0.8:
            label = 0
        elif pga_value < 2.5:
            label = 1
        elif pga_value < 8.0:
            label = 2
        elif pga_value < 25:
            label = 3
        elif pga_value < 80:
            label = 4
        else:
            label = 5

        waveform = None
        for year in self.h5_file.keys():
            if trace_name in self.h5_file[year]:
                waveform = np.array(self.h5_file[f"{year}/{trace_name}"])
                break

        if waveform is None:
            raise ValueError(f"Waveform {trace_name} not found in HDF5 file.")

        start_idx = p_arrival
        end_idx = start_idx + 300

        if end_idx > waveform.shape[1]:
            raise ValueError(f"Trace {trace_name} 不足 300 samples，請確認數據！")

        waveform = waveform[:, start_idx:end_idx]
        waveform = torch.tensor(waveform, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label


def create_dataloaders(hdf5_path, metadata_csv, batch_size=32, split_ratio=0.67, random_seed=42):
    """
    建立訓練/驗證 DataLoader，並保證 trace_pga_cmps2 分布平衡
    """
    df = pd.read_csv(metadata_csv)
    df = df[
        (df["station_location_code"] == 10) &
        (df["trace_snr_db"] >= 10) &
        (df["trace_p_arrival_sample"] >= 500) &
        (df["trace_channel"].isin(["HN", "HL"])) &
        (df["path_ep_distance_km"] <= 100) &
        (df["trace_completeness"] >= 3)
    ]

    df["pga_bin"] = pd.qcut(df["trace_pga_cmps2"], q=10, duplicates='drop')

    train_idx, val_idx = train_test_split(
        df.index.values,
        train_size=split_ratio,
        stratify=df["pga_bin"],
        random_state=random_seed
    )

    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)

    train_dataset = CWADataset(hdf5_path, train_df)
    val_dataset = CWADataset(hdf5_path, val_df)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        prefetch_factor=4, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
        prefetch_factor=4, persistent_workers=True
    )

    print(f"✅ 資料分割完成：訓練集 {len(train_dataset)} 筆，驗證集 {len(val_dataset)} 筆")
    return train_loader, val_loader

def load_test_dataloader(hdf5_path, metadata_csv, batch_size=32):
    """
    直接載入測試用 DataLoader，不做任何分割
    """
    df = pd.read_csv(metadata_csv)

    # 如果你 test 也要過濾（可選）
    df = df[
        (df["station_location_code"] == 10) &
        (df["trace_snr_db"] >= 10) &
        (df["trace_p_arrival_sample"] >= 500) &
        (df["trace_channel"].isin(["HN", "HL"])) &
        (df["path_ep_distance_km"] <= 100) &
        (df["trace_completeness"] >= 3)
    ]

    dataset = CWADataset(hdf5_path, df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print(f"🧪 測試集載入完成，共 {len(dataset)} 筆")
    return dataloader


def show_class_distribution(dataset, name="Dataset"):
    labels = [label for _, label in dataset]
    labels = torch.tensor(labels)
    total = len(labels)

    level_names = [
        "0級(無感)<0.8",
        "1級(微震)0.8~2.5",
        "2級(輕震)2.5~8.0",
        "3級(弱震)8.0~25",
        "4級(中震)25~80",
        "5級(強震)80~250",
        "6級(烈震)250~400",
        "7級(毀震)>400"
    ]

    print(f"📊 {name} 類別分布：")
    for i in range(8):
        count = (labels == i).sum().item()
        print(f"  {i}：{level_names[i]} → {count} 筆 ({count / total:.2%})")
    print()


def plot_class_distribution(dataset, name="Dataset"):
    labels = [label for _, label in dataset]
    labels = torch.tensor(labels)

    level_names = [
        "0 intensity\n",
        "1 intensity\n",
        "2 intensity\n",
        "3 intensity\n",
        "4 intensity\n",
        "5 intensity\n",
        "6 intensity\n",
        "7 intensity\n"
    ]

    counts = [(labels == i).sum().item() for i in range(8)]

    print(f"📊 {name} 類別分布：")
    for i, count in enumerate(counts):
        print(f"  {i}：{level_names[i]} → {count} 筆 ({count / len(labels):.2%})")

    plt.figure(figsize=(10, 6))
    bars = plt.bar(level_names, counts, color='orange')
    plt.title(f"{name}  PGA Distribution", fontsize=16)
    plt.xlabel("Intensity", fontsize=12)
    plt.ylabel("Number", fontsize=12)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f'{yval}', ha='center', va='bottom')

    plt.tight_layout()
    os.makedirs("plot", exist_ok=True)
    plt.savefig(f"plot/{name}_class_distribution.png")
    plt.close()


if __name__ == "__main__":
    metadata_path = "CWA_processed_data/all_metadata.csv"
    hdf5_path = "CWA_processed_data/all.hdf5"

    dataset_df = pd.read_csv(metadata_path)
    new_dataset = CWADataset(hdf5_path, dataset_df)
    print(f"📁 資料集大小: {len(new_dataset)}")


    dataset_df = pd.read_csv(metadata_path)
    dataset_df = dataset_df[
        (dataset_df["station_location_code"] == 10) &
        (dataset_df["trace_snr_db"] >= 0) &
        (dataset_df["trace_p_arrival_sample"] >= 500) &
        (dataset_df["trace_channel"].isin(["HN", "HL"])) &
        (dataset_df["path_ep_distance_km"] <= 270) &
        (dataset_df["trace_completeness"] >= 3)
    ]

    new_dataset = CWADataset(hdf5_path, dataset_df)
    print(f"📁 資料集大小: {len(new_dataset)}")
    # show_class_distribution(new_dataset, "Test Dataset")
    # plot_class_distribution(new_dataset, "Test Dataset")

    dataset_df = pd.read_csv(metadata_path)
    dataset_df = dataset_df[
        (dataset_df["station_location_code"] == 10) &
        (dataset_df["trace_snr_db"] >= 10) &
        (dataset_df["trace_p_arrival_sample"] >= 500) &
        (dataset_df["trace_channel"].isin(["HN", "HL"])) &
        (dataset_df["path_ep_distance_km"] <= 100) &
        (dataset_df["trace_completeness"] >= 3)
    ]

    old_dataset = CWADataset(hdf5_path, dataset_df)
    print(f"📁 資料集大小: {len(old_dataset)}")
    # show_class_distribution(old_dataset, "Test Dataset")
    # plot_class_distribution(old_dataset, "Test Dataset")