import os
import numpy as np
import pandas as pd
import h5py
from scipy.signal import butter, filtfilt

# ---------------------------
# 1. 基本設定
# ---------------------------

# 原始資料所在資料夾（所有 HDF5 與 CSV 檔案都放在此資料夾中）
DATA_PATHS = r"Y:\CWB24\CWASN"  # 使用原始字串避免路徑轉譯問題

# 定義要處理的年份（不再區分 train/valid/test）
YEARS = [2012, 2019, 2020, 2021]

# 設定輸出資料夾及檔名
OUTPUT_DIR = "CWA_processed_data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_HDF5 = os.path.join(OUTPUT_DIR, "all.hdf5")      # 處理後數據存成一個 HDF5 檔
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "all_metadata.csv") # 合併後的 metadata 儲存為 CSV 檔

# ---------------------------
# 2. 低通濾波器函數定義
# ---------------------------
def butter_lowpass_filter(data, cutoff=10, fs=100, order=4):
    """
    對輸入的三軸數據 (shape: (3, samples)) 使用 Butterworth 低通濾波器進行濾波處理
      - cutoff: 截止頻率 (Hz)
      - fs: 取樣頻率 (Hz)
      - order: 濾波器階數
    若資料中含有 NaN 或 Inf 則直接使用原始數據
    """
    nyq = 0.5 * fs                 # 計算奈奎斯特頻率
    normal_cutoff = cutoff / nyq   # 正規化截止頻率
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    filtered = np.zeros_like(data)  # 建立一個與原始資料相同 shape 的陣列存放結果
    # 分別對三個通道 (例如 Z, N, E) 執行濾波
    for i in range(3):
        channel = data[i, :]
        # 如果資料中有 NaN 或 Inf 則不做濾波，直接複製原資料
        if np.isnan(channel).any() or np.isinf(channel).any():
            print(f"警告：通道 {i} 含有 NaN 或 Inf，跳過濾波")
            filtered[i, :] = channel
            continue
        try:
            # 只有在資料長度足夠時才執行濾波，否則直接傳回原資料
            if len(channel) >= 15:
                filtered[i, :] = filtfilt(b, a, channel, padlen=9)
            else:
                print(f"警告：通道 {i} 資料長度不足，跳過濾波")
                filtered[i, :] = channel
        except Exception as e:
            print(f"濾波失敗：通道 {i} 出錯 ({e})，使用原始資料")
            filtered[i, :] = channel
    return filtered

# ---------------------------
# 3. 資料整合及處理主流程
# ---------------------------
def process_all_data():
    # 用來儲存所有年份的 metadata，之後合併成一個 CSV 輸出
    metadata_list = []
    
    # 開啟輸出 HDF5 檔案，將所有年份處理後的波形數據存入此檔案
    with h5py.File(OUTPUT_HDF5, "w") as hdf5_out:
        # 遍歷每個年份
        for year in YEARS:
            # 組成該年份的 HDF5 檔及 metadata CSV 檔路徑
            hdf5_file = os.path.join(DATA_PATHS, f"chunks_{year}.hdf5")
            metadata_file = os.path.join(DATA_PATHS, f"metadata_{year}.csv")
            
            # 若任一檔案不存在，則略過此年份
            if not os.path.exists(hdf5_file) or not os.path.exists(metadata_file):
                print(f"檔案不存在：{year}，跳過")
                continue
            
            print(f"處理 {year} ...")
            
            # 讀取該年份的 metadata 並新增一個年份欄位
            meta = pd.read_csv(metadata_file)
            meta["year"] = year
            
            # 篩選符合條件的記錄：
            #   1. source_event_id 必須以 "0" 結尾
            #   2. station_location_code 必須為 10
            #   3. trace_channel 必須是 "HN" 或 "HL"
            #   4. trace_p_arrival_sample 必須大於等於 500
            meta = meta[
                (meta["source_event_id"].astype(str).str.endswith("0")) &
                (meta["station_location_code"] == 10) &
                (meta["trace_channel"].isin(["HN", "HL"])) &
                (meta["trace_p_arrival_sample"] >= 500)
            ]
            if meta.empty:
                print(f"{year} 中無符合條件的資料，跳過")
                continue
            
            # 將符合條件的 metadata 加入總清單
            metadata_list.append(meta)
            
            # 開啟該年份的原始 HDF5 檔案，取得 "data" 群組中的波形數據
            with h5py.File(hdf5_file, "r") as hdf5_in:
                data_group = hdf5_in["data"]
                # 遍歷該年份每一筆 metadata 記錄
                for _, row in meta.iterrows():
                    trace_name = row["trace_name"]
                    # 若原始檔案中找不到該 trace 則略過
                    if trace_name not in data_group:
                        print(f"注意：{trace_name} 不在檔案中，跳過")
                        continue
                    # 讀取該筆波形資料（預期 shape 為 (3, samples)）
                    waveform = data_group[trace_name][()]
                    
                    # 數據單位轉換：根據 sensitivity 參數將數值換算成對應的物理量
                    sens_str = row["station_sensitivity_counts_spm"]
                    sensitivity = np.array(sens_str.replace(",", "").strip("[]").split()).astype(float)
                    waveform *= sensitivity.reshape(3, 1)
                    
                    # 去均值：消除直流偏移量
                    waveform -= np.mean(waveform, axis=1, keepdims=True)
                    
                    # 低通濾波：以 10 Hz 為截止頻率進行濾波
                    waveform = butter_lowpass_filter(waveform)
                    
                    # 定義唯一的 dataset 名稱，格式為 "year_trace_name"
                    dataset_name = f"{year}_{trace_name}"
                    # 若 HDF5 中已存在此名稱的資料，先刪除再新增
                    if dataset_name in hdf5_out:
                        del hdf5_out[dataset_name]
                    hdf5_out.create_dataset(dataset_name, data=waveform)
            
            print(f"{year} 處理完成")
    
    # 將所有年份的 metadata 合併後儲存成一個 CSV 檔案
    if metadata_list:
        merged_meta = pd.concat(metadata_list, ignore_index=True)
        merged_meta.to_csv(OUTPUT_CSV, index=False)
        print(f"所有 metadata 已儲存至：{OUTPUT_CSV}")
    
    print(f"所有數據處理完成，結果儲存在：{OUTPUT_HDF5}")


# ---------------------------
# 程式進入點
# ---------------------------
if __name__ == "__main__":
    process_all_data()
