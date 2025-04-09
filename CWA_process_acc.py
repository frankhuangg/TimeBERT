import os
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm

# -------------------------------
# 單一資料集處理函式 (例如: train, test, valid)
# -------------------------------
def process_dataset(dataset_type, base_path):
    """
    根據指定資料類型（如 train、test、valid），讀取對應的 CSV 與 HDF5 檔案，
    篩選出符合條件的事件後，
    將加速度波形 (直接從 HDF5 取得、已是加速度波形) 並 trim 至 6000 點，
    同時收集部分 metadata 整理成 numpy 陣列回傳。
    """
    csv_file = os.path.join(base_path, f"{dataset_type}.csv")
    hdf5_file = os.path.join(base_path, f"{dataset_type}.hdf5")
    print(f"\n📂 Processing: {csv_file}, {hdf5_file}")

    # 讀取 CSV 並依條件篩選 (請根據實際 CSV 欄位調整)
    df = pd.read_csv(csv_file)
    df = df[(df.trace_category == 'earthquake_local') &
            (df.source_distance_km <= 20) &
            (df.source_magnitude >= 3)]
    print(f"✅ Selected {len(df)} events from {dataset_type}.csv")

    # 初始化儲存結果的 list
    waveforms_list = []
    metadata_list = []
    
    # 開啟 HDF5 檔案，假設資料儲存在 "data/{trace_name}" 路徑下
    dtfl = h5py.File(hdf5_file, 'r')
    
    # 統計用的計數器
    skipped_no_dataset = 0
    skipped_too_short = 0
    processed_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=dataset_type):
        trace_name = row['trace_name']
        dataset_path = f"data/{trace_name}"
        if dataset_path not in dtfl:
            skipped_no_dataset += 1
            continue
        
        dataset = dtfl[dataset_path]
        data = np.array(dataset)
        # 如果資料點數不足 6000 則略過該筆資料
        if data.shape[0] < 6000:
            skipped_too_short += 1
            continue
        
        # 取得前 6000 點數值，轉置成 shape (3, 6000)
        processed_data = data[:6000, :].T
        
        # 收集部分 metadata (請依據你 CSV 欄位修改)
        meta = [
            row['source_distance_km'],
            row['source_depth_km'],
            row['source_magnitude'],
            row['receiver_latitude'],
            row['receiver_longitude'],
            row['source_latitude'],
            row['source_longitude']
        ]
        waveforms_list.append(processed_data)
        metadata_list.append(meta)
        processed_count += 1

    dtfl.close()
    print(f"📊 {dataset_type}: 總共 {len(df)} 筆, 處理 {processed_count} 筆, "
          f"跳過 (無 dataset: {skipped_no_dataset}, 太短: {skipped_too_short})")

    if not waveforms_list:
        raise ValueError(f"{dataset_type} 沒有任何資料被處理，請檢查輸入條件。")
    
    # 整理結果，waveforms shape 為 (N, 3, 6000)，metadata shape 為 (N, 7)
    waveforms = np.stack(waveforms_list, axis=0)
    metadata = np.array(metadata_list, dtype=np.float32)
    return waveforms, metadata

# -------------------------------
# 主程式：依序處理各資料集並儲存結果
# -------------------------------
def main():
    base_path = r"Z:\seismic\STEAD"  # 請依據實際資料夾路徑修改
    dataset_types = ['train', 'test', 'valid']

    for ds in dataset_types:
        try:
            waveforms, metadata = process_dataset(ds, base_path)
            output_file = os.path.join(base_path, f"{ds}_acceleration.npz")
            np.savez(output_file, waveforms=waveforms, metadata=metadata)
            print(f"\n✅ Saved {waveforms.shape[0]} acceleration waveforms to {output_file}")
        except Exception as e:
            print(f"❌ 處理 {ds} 時發生錯誤: {e}")

if __name__ == '__main__':
    main()
