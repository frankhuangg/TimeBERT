# cwa_dataloader.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import atexit # 用於確保程序退出時關閉文件

# !!! 重要：請根據您的分析選擇 8 個 CWA 屬性 !!!
# 範例選擇 (需要您確認並修改，特別是 S 波和替代 coda 的屬性):
SELECTED_CWA_ATTRS = [
    'trace_p_arrival_sample',
    'trace_p_weight',         # 確認 TSMIP 是否有/如何處理
    'path_p_travel_s',
    'trace_s_arrival_sample', # 確認 TSMIP 是否有/如何處理
    'trace_s_weight',         # 確認 TSMIP 是否有/如何處理
    'path_ep_distance_km',
    'path_back_azimuth_deg',
    'source_magnitude'        # 使用震級替代 coda_end_sample
]

# --- Helper function for DataLoader ---
def collate_fn_skip_none(batch):
    """ Filters out None items from the batch and collates the rest. """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None
    try:
        # 返回的是 waveform segment, info
        info = torch.stack([item[0] for item in batch]) # item[0] 是 info
        waveform = torch.stack([item[1] for item in batch]) # item[1] 是 waveform
        return info, waveform
    except Exception as e:
         print(f"Error during batch stacking in collate_fn: {e}")
         return None, None

class cwa_loader(Dataset):
    def __init__(self, metadata_csv_path, hdf5_path, segment_length=2000, p_arrival_offset=500):
        """
        Args:
            metadata_csv_path (str): 處理後的 metadata CSV 文件路徑 (e.g., train.csv).
            hdf5_path (str): 處理後的 HDF5 文件路徑 (e.g., train.hdf5).
            segment_length (int): 截取的波形段長度 (樣本點數).
            p_arrival_offset (int): P波到達時間在截取窗口中的位置.
        """
        self.metadata_path = metadata_csv_path
        self.hdf5_path = hdf5_path
        self.segment_length = segment_length
        self.p_arrival_offset = p_arrival_offset
        if p_arrival_offset >= segment_length:
            raise ValueError("p_arrival_offset must be less than segment_length")

        try:
            self.metadata = pd.read_csv(self.metadata_path)
            print(f"成功從 {self.metadata_path} 加載 {len(self.metadata)} 筆 metadata。")
            required_cols = ['trace_name', 'year', 'trace_p_arrival_sample'] + SELECTED_CWA_ATTRS
            missing_cols = [col for col in required_cols if col not in self.metadata.columns]
            if missing_cols:
                raise ValueError(f"Metadata 文件缺少必要欄位: {missing_cols}")
        except Exception as e:
            print(f"讀取 Metadata 文件時發生錯誤: {e}")
            raise

        # --- 在 __init__ 中打開 HDF5 文件 ---
        self.hdf5_file = None # 初始化為 None
        try:
            # 使用 'r' 模式打開文件，libver='latest' 可能有助於性能
            self.hdf5_file = h5py.File(self.hdf5_path, 'r', libver='latest')
            print(f"成功打開 HDF5 文件: {self.hdf5_path}")
            # 註冊一個函數，在程序退出時關閉文件
            atexit.register(self._close_hdf5)
        except Exception as e:
            print(f"打開 HDF5 文件 {self.hdf5_path} 時發生錯誤: {e}")
            self.hdf5_file = None # 確保如果打開失敗，句柄是 None
            raise # 重新拋出錯誤，因為沒有 HDF5 無法工作

    def _close_hdf5(self):
        # 用於 atexit 註冊的關閉函數
        if self.hdf5_file:
            try:
                print(f"正在關閉 HDF5 文件: {self.hdf5_path}")
                self.hdf5_file.close()
            except Exception as e:
                print(f"關閉 HDF5 文件時出錯: {e}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # --- 檢查 HDF5 文件句柄是否存在 ---
        if self.hdf5_file is None:
             print("錯誤: HDF5 文件未成功打開。")
             return None # 或者引發異常
        
        meta_row = self.metadata.iloc[idx]
        year = meta_row["year"]
        trace_name = meta_row["trace_name"]

        # --- 使用已打開的文件句柄讀取 HDF5 ---
        try:
            hdf5_datapath = f"{year}/{trace_name}"
            # 檢查路徑是否存在
            if hdf5_datapath not in self.hdf5_file:
                 # print(f"Warning: {hdf5_datapath} not in HDF5 file. Skipping.")
                 return None
            waveform_data = self.hdf5_file[hdf5_datapath][()]
        except Exception as e:
            # print(f"Error reading {hdf5_datapath}: {e}")
            return None


        # --- 截取 20 秒波形段 ---
        p_arrival = int(meta_row['trace_p_arrival_sample'])
        start_index = p_arrival - self.p_arrival_offset
        end_index = start_index + self.segment_length

        # 使用 NumPy 更高效地處理邊界和填充
        segment = np.zeros((3, self.segment_length), dtype=waveform_data.dtype)
        src_start = max(0, start_index)
        src_end = min(waveform_data.shape[1], end_index)
        dst_start = max(0, -start_index)
        length_to_copy = src_end - src_start

        # 確保有合法的數據可以複製
        if length_to_copy > 0 and dst_start < self.segment_length:
            # 計算實際能放入目標 segment 的長度
            actual_copy_len = min(length_to_copy, self.segment_length - dst_start)
            segment[:, dst_start : dst_start + actual_copy_len] = waveform_data[:, src_start : src_start + actual_copy_len]
        elif length_to_copy <= 0 and dst_start >= self.segment_length :
             # This case handles scenarios where the calculated window is entirely outside the data
             # or the source data itself is empty for the needed range.
             # It might be okay if the segment remains zeros, or return None if invalid
             # print(f"Warning: No valid data to copy for trace {trace_name}")
             pass # Keep segment as zeros, or return None

        # --- 處理事件資訊 ---
        try:
            selected_info = meta_row[SELECTED_CWA_ATTRS]
            info_normalized = self.norm_text_cwa(selected_info)
            # 檢查 norm_text_cwa 是否返回了 NaN
            if np.isnan(info_normalized).any():
                # print(f"Warning: NaN found in normalized info for trace {trace_name}. Skipping.")
                return None
        except Exception as e:
            # print(f"Error processing info for trace {trace_name}: {e}")
            return None

        # --- 轉換為 Tensor ---
        waveform_tensor = torch.tensor(segment, dtype=torch.float32)
        info_tensor = torch.tensor(info_normalized, dtype=torch.float32)

        return info_tensor, waveform_tensor

    def norm_text_cwa(self, selected_info):
        """
        對選定的 CWA 屬性進行歸一化 (基於 train.csv 統計數據)。
        """
        y = np.array(selected_info.values, dtype='float')
        expected_len = len(SELECTED_CWA_ATTRS)

        if len(y) != expected_len:
             # print(f"Warning: Initial info length {len(y)} != expected {expected_len}. Returning NaN array.")
             return np.full(expected_len, np.nan, dtype=np.float32) # 返回 NaN

        # 標記 NaN
        mask = np.isnan(y)
        y[mask] = -1.0

        try:
            # 1. trace_p_arrival_sample: 99% 約 9249, Max 約 25748. 使用 10000 作為上限歸一化。
            p_arrival_norm_max = 10000.0
            if y[0] != -1: y[0] = np.clip(y[0] / p_arrival_norm_max, 0, 1)

            # 2. trace_p_weight: 範圍 0-4. 除以 4.0 歸一化到 [0, 1]。
            if y[1] != -1: y[1] = np.clip(y[1] / 4.0, 0, 1)

            # 3. path_p_travel_s: 99% 約 36.33, Max 約 80.75. 使用 40.0 作為上限歸一化。
            p_travel_norm_max = 40.0
            if y[2] != -1: y[2] = np.clip(y[2] / p_travel_norm_max, 0, 1)

            # 4. trace_s_arrival_sample: 99% 約 10719, Max 約 26574. 使用 11000 作為上限歸一化。
            s_arrival_norm_max = 11000.0
            if y[3] != -1: y[3] = np.clip(y[3] / s_arrival_norm_max, 0, 1)

            # 5. trace_s_weight: 範圍 0-4 (根據數據). 除以 4.0 歸一化到 [0, 1]。
            if y[4] != -1: y[4] = np.clip(y[4] / 4.0, 0, 1)

            # 6. path_ep_distance_km: 99% 約 266, Max 約 618. 使用 300.0 作為上限歸一化。
            dist_norm_max = 300.0
            if y[5] != -1: y[5] = np.clip(y[5] / dist_norm_max, 0, 1)

            # 7. path_back_azimuth_deg: 範圍 0-360. 除以 360.0 歸一化到 [0, 1]。
            if y[6] != -1: y[6] = np.clip(y[6] / 360.0, 0, 1)

            # 8. source_magnitude: Min 1.54, Max 6.91. 使用觀察到的 Min/Max 進行歸一化。
            min_mag, max_mag = 1.54, 6.91
            if y[7] != -1:
                 if max_mag > min_mag: # 避免除以零
                     y[7] = (y[7] - min_mag) / (max_mag - min_mag)
                 else:
                     y[7] = 0.0 # 如果 min == max，設為 0
                 y[7] = np.clip(y[7], 0, 1) # 截斷到 [0, 1]

        except IndexError:
             # print(f"Error: Index out of bounds during normalization.")
             return np.full(expected_len, np.nan, dtype=np.float32)

        # 將之前標記的 -1 (代表原始 NaN) 替換為 0.0
        y[y == -1.0] = 0.0

        # 最終檢查 NaN (如果歸一化計算產生問題)
        if np.isnan(y).any():
             # print(f"Warning: Final normalized info contains NaN. Replacing with 0.")
             y[np.isnan(y)] = 0.0 # 再次確保沒有 NaN

        return y.astype(np.float32)

# --- 用於測試 Dataloader 的代碼 ---
if __name__ == '__main__':
    # --- (與上次相同，用於測試 dataloader 能否運行) ---
    CWA_METADATA_TRAIN_PATH = "D:/shihan/pretrain_data_all_channels_accel/train.csv"
    CWA_HDF5_TRAIN_PATH = "D:/shihan/pretrain_data_all_channels_accel/train.hdf5"

    if not os.path.exists(CWA_METADATA_TRAIN_PATH) or not os.path.exists(CWA_HDF5_TRAIN_PATH):
        print("測試文件缺失")
    else:
        print("測試 CWA Dataloader (返回波形段)...")
        try:
            train_dataset = cwa_loader(
                metadata_csv_path=CWA_METADATA_TRAIN_PATH,
                hdf5_path=CWA_HDF5_TRAIN_PATH,
                segment_length=2000,
                p_arrival_offset=500
            )
            print(f"Dataset size: {len(train_dataset)}")
            if len(train_dataset) > 0:
                # ... (查找並打印第一個有效樣本 - 與上次相同) ...
                first_valid_sample = None
                # ... (查找循環) ...
                if first_valid_sample:
                    # ... (打印樣本 shape) ...
                    print("\n測試 DataLoader...")
                    # ... (創建和測試 DataLoader - 與上次相同) ...

        except Exception as e: print(f"初始化 Dataloader 時錯誤: {e}")