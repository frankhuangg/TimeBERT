import os
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

class CWADataset(Dataset):
    def __init__(self, hdf5_path, metadata, transform=None):
        """
        初始化 CWA 數據集（metadata 為已過濾的 DataFrame）
        - hdf5_path: HDF5 數據檔案的路徑，內含波形資料
        - metadata: CSV 路徑或 pandas DataFrame，記錄數據屬性與參數
        - transform: 額外的轉換操作（如 normalization, augmentation 等）
        """
        self.hdf5_path = hdf5_path
        self.h5_file = None  # 採用 lazy open 模式，第一次訪問 __getitem__ 時才打開
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

        # 將取得的 trace_name 轉換為 Python 字串
        trace_name = str(self.trace_names[idx])
        p_arrival = self.p_arrival_samples[idx]
        pga_value = self.pga_values[idx]

        # 設定分類標籤（根據 trace_pga_cmps2 的數值）
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

        # 由於所有波形數據直接存放在根目錄中，其名稱格式為 "year_trace_name"
        # 我們遍歷所有 dataset 名稱，找出符合結尾為 "_{trace_name}" 的 dataset
        found_key = None
        for key in self.h5_file.keys():
            if key.endswith(f"_{trace_name}"):
                found_key = key
                break

        if found_key is None:
            raise ValueError(f"Waveform {trace_name} not found in HDF5 file.")

        waveform = np.array(self.h5_file[found_key])

        # 假設採樣率 fs 為 100 Hz
        fs = 100
        pre_samples = 5 * fs   # 提取 P 波前 5 秒，即 500 samples
        post_samples = 25 * fs  # 提取 P 波後 25 秒，即 2500 samples

        # 計算起始與結束 index
        start_idx = p_arrival - pre_samples
        end_idx = p_arrival + post_samples

        # 檢查數據長度，確保不超出範圍
        if start_idx < 0 or end_idx > waveform.shape[1]:
            raise ValueError(f"Trace {trace_name} 的數據長度不足以提取從 P 波前 5 秒到 P 波後 25 秒的數據！")

        # 提取出目標波形片段，shape 將為 (3, 3000)
        waveform = waveform[:, start_idx:end_idx]
        waveform = torch.tensor(waveform, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label



def extract_waveforms_from_hdf5(hdf5_path, metadata_df):
    """
    根據 metadata 中的 trace_name 從 HDF5 中提取 P-wave 附近的波形區段
    - input: metadata_df - 包含 trace_name 和 p_arrival 資訊的 DataFrame
    - return: (N, 3, 3000) 的 waveform_array, 和 metadata_array
    """

    waveforms = []
    metadata_records = []
    pre_sec=5
    post_sec=25
    fs=100
    candidate_columns = [
        "station_code",
        "trace_pga_cmps2",
        "station_latitude_deg",
        "station_longitude_deg",
        "source_latitude_deg",
        "source_longitude_deg",
        "source_depth_km",
        "path_ep_distance_km"
    ]
    selected_columns = [col for col in candidate_columns if col in metadata_df.columns]

    numeric_cols = {
        "trace_pga_cmps2",
        "station_latitude_deg",
        "station_longitude_deg",
        "source_latitude_deg",
        "source_longitude_deg",
        "source_depth_km",
        "path_ep_distance_km"
    }

    with h5py.File(hdf5_path, 'r') as h5:
        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            trace_name = row['trace_name']
            p_arrival = int(row['trace_p_arrival_sample'])

            found_key = next((k for k in h5.keys() if k.endswith(f"_{trace_name}")), None)
            if found_key is None:
                continue

            data = np.array(h5[found_key])
            start = p_arrival - pre_sec * fs
            end = p_arrival + post_sec * fs

            if start < 0 or end > data.shape[1]:
                continue

            waveform = data[:, start:end]
            if waveform.shape[1] == (pre_sec + post_sec) * fs:
                waveforms.append(waveform)
                values = []
                for col in selected_columns:
                    val = row[col]
                    if col in numeric_cols:
                        values.append(np.float32(val))
                    else:
                        values.append(val)
                metadata_records.append(values)
    if not waveforms:
        raise RuntimeError("No valid waveforms extracted.")

    waveform_array = np.stack(waveforms, axis=0)
    metadata_array = pd.DataFrame(metadata_records, columns=selected_columns)
    return waveform_array, metadata_array


def create_dataloaders_7_2_1(hdf5_path, metadata_csv, batch_size=32, random_seed=42):
    """
    建立訓練、測試與驗證的 DataLoader，比例為 7 : 2 : 1
      - 從 metadata_csv 讀入資料並過濾（例如 station_location_code、trace_snr_db、trace_p_arrival_sample 等條件）
      - 使用 pd.qcut 對 trace_pga_cmps2 進行分箱，作為分層抽樣依據
      - 第一次分割：70% 資料作訓練，剩下 30% 作為暫存集合 (temp)
      - 第二次分割：在暫存集合中以 2:1 分割出測試 (約 20%) 與驗證 (約 10%)
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

    # 以 trace_pga_cmps2 進行分箱，用於 stratify 分層抽樣
    df["pga_bin"] = pd.qcut(df["trace_pga_cmps2"], q=10, duplicates="drop")

    # 第一次分割：70% 資料作訓練，剩下 30% 作為暫存集合
    train_idx, temp_idx = train_test_split(
        df.index.values,
        train_size=0.7,
        stratify=df["pga_bin"],
        random_state=random_seed
    )
    # 第二次分割：在暫存集合中以 2:1 分出測試和驗證集
    test_idx, val_idx = train_test_split(
        temp_idx,
        train_size=2 / 3,
        stratify=df.loc[temp_idx, "pga_bin"],
        random_state=random_seed
    )

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)

    print("\n📦 Extracting training waveforms")
    train_wave, train_meta = extract_waveforms_from_hdf5(hdf5_path, train_df)
    print("📦 Extracting testing waveforms")
    test_wave, test_meta = extract_waveforms_from_hdf5(hdf5_path, test_df)
    print("📦 Extracting validation waveforms")
    val_wave, val_meta = extract_waveforms_from_hdf5(hdf5_path, val_df)

    return (train_wave, train_meta), (test_wave, test_meta), (val_wave, val_meta)
    # # 波形資料
    # train_dataset = CWADataset(hdf5_path, train_df)
    # test_dataset = CWADataset(hdf5_path, test_df)
    # val_dataset = CWADataset(hdf5_path, val_df)

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=4,
    #     pin_memory=True,
    #     prefetch_factor=4,
    #     persistent_workers=True,
    # )
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True,
    #     prefetch_factor=4,
    #     persistent_workers=True,
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True,
    #     prefetch_factor=4,
    #     persistent_workers=True,
    # )

    # print(
    #     f"✅ 資料分割完成：訓練集 {len(train_dataset)} 筆，"
    #     f"測試集 {len(test_dataset)} 筆，驗證集 {len(val_dataset)} 筆"
    # )
    # return train_loader, test_loader, val_loader


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
    bars = plt.bar(level_names, counts, color="orange")
    plt.title(f"{name}  PGA Distribution", fontsize=16)
    plt.xlabel("Intensity", fontsize=12)
    plt.ylabel("Number", fontsize=12)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval}", ha="center", va="bottom")

    plt.tight_layout()
    os.makedirs("plot", exist_ok=True)
    plt.savefig(f"plot/{name}_class_distribution.png")
    plt.close()

def plot_sample_waveform(dataset, idx=0, fs=100, save_path = None):
    """
    從給定的 dataset 中取出第 idx 筆數據，並繪製波形
      - dataset: 已經建立好的 CWADataset 物件
      - idx: 要查看的數據索引（預設 0）
      - fs: 採樣頻率，預設為 100 Hz
    提取的波形為從 P 波到達 sample 開始連續 300 個 samples，
    對於 100 Hz 的數據，此範圍大約代表 3 秒
    """
    # 取得數據集中的第 idx 筆數據
    waveform = dataset[idx]  
    # 將 tensor 轉為 numpy 陣列，形狀應為 (3, samples)
    waveform_np = waveform.numpy()
    samples = waveform_np.shape[1]
    # 依照採樣頻率生成時間軸（單位秒）
    t = np.arange(samples) / fs

    # 為三個軸分別建立子圖
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # 預設的通道標籤，可依據實際情況修改
    channel_labels = ["Z", "N", "E"]
    for i in range(3):
        axs[i].plot(t, waveform_np[i])
        axs[i].set_title(f"Channel {i} ({channel_labels[i]})")
        axs[i].set_ylabel("Amplitude")
        axs[i].grid(True)
    
    axs[-1].set_xlabel("Time (s)")
    plt.suptitle(f"Extracted Waveform ", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()
    # 如果指定了 save_path，則存檔；否則顯示圖像
    if save_path is not None:
        # 如果資料夾不存在，建立它
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    plt.close()

def get_num_domains(metadata_csv):
    df = pd.read_csv(metadata_csv)
    df = df[
        (df["station_location_code"] == 10) &
        (df["trace_snr_db"] >= 10) &
        (df["trace_p_arrival_sample"] >= 500) &
        (df["trace_channel"].isin(["HN", "HL"])) &
        (df["path_ep_distance_km"] <= 100) &
        (df["trace_completeness"] >= 3)
    ]
    
    if "station_code" in df.columns:
        station_ids = df["station_code"]
    elif "station_latitude_deg" in df.columns and "station_longitude_deg" in df.columns:
        station_ids = df["receiver_latitude"].astype(str) + "_" + df["receiver_longitude"].astype(str)
    else:
        raise ValueError("找不到 station_code 或經緯度欄位")

    return station_ids.nunique()

if __name__ == "__main__":
    metadata_path = "CWA_processed_data/all_metadata.csv"
    hdf5_path = "CWA_processed_data/all.hdf5"

    # 使用 create_dataloaders_7_2_1 分割成訓練、測試與驗證集（比例 7:2:1）
    (train_wave, train_meta), (test_wave, test_meta), (val_wave, val_meta) = create_dataloaders_7_2_1(hdf5_path, metadata_path)

    # 調用上面定義的繪圖函數，顯示索引 0 的波形
    save_path = "plot/sample_waveform.png"
    plot_sample_waveform(train_wave, idx=0, fs=100, save_path = save_path)
