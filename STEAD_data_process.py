import os
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client
import matplotlib.pyplot as plt
from obspy.clients.fdsn.header import FDSNNoDataException

# -------------------------------
# ObsPy 轉換函式：將 HDF5 dataset 轉成 ObsPy stream
# -------------------------------
def make_stream(dataset):
    data = np.array(dataset)

    tr_E = obspy.Trace(data=data[:, 0])
    tr_E.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_E.stats.delta = 0.01
    tr_E.stats.channel = dataset.attrs['receiver_type'] + 'E'
    tr_E.stats.station = dataset.attrs['receiver_code']
    tr_E.stats.network = dataset.attrs['network_code']

    tr_N = obspy.Trace(data=data[:, 1])
    tr_N.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_N.stats.delta = 0.01
    tr_N.stats.channel = dataset.attrs['receiver_type'] + 'N'
    tr_N.stats.station = dataset.attrs['receiver_code']
    tr_N.stats.network = dataset.attrs['network_code']

    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.channel = dataset.attrs['receiver_type'] + 'Z'
    tr_Z.stats.station = dataset.attrs['receiver_code']
    tr_Z.stats.network = dataset.attrs['network_code']

    return obspy.Stream([tr_E, tr_N, tr_Z])

def make_plot(tr, title='', ylab=''):
    """
    繪製單一 trace 的圖形
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(tr.times("matplotlib"), tr.data, "k-")
    ax.xaxis_date()
    fig.autofmt_xdate()
    plt.ylabel(ylab)
    plt.title(title)
    plt.show()

# -------------------------------
# 資料處理：讀取各 chunk、轉換成加速度並儲存
# -------------------------------
def process_chunks(chunk_ids, base_path):
    """
    依據指定 chunk 讀取 CSV 與 HDF5 檔，
    篩選出地震區域事件（source_distance_km <= 20、source_magnitude >= 3），
    並利用 IRIS 取得儀器響應，將波形轉換成加速度，
    最後將每筆波形 trim 至 6000 點後回傳與 metadata。
    """
    client = Client("IRIS")
    inventory_cache = {}
    all_waveforms = []
    all_metadata = []

    for cid in chunk_ids:
        csv_file = os.path.join(base_path, f"chunk{cid}.csv")
        hdf5_file = os.path.join(base_path, f"chunk{cid}.hdf5")
        print(f"\n📂 Processing: {csv_file}, {hdf5_file}")
        
        # 讀取 CSV 並依條件篩選
        df = pd.read_csv(csv_file)
        df = df[(df.trace_category == 'earthquake_local') &
                (df.source_distance_km <= 20) & 
                (df.source_magnitude >= 3)]
        print(f"✅ Selected {len(df)} events from chunk{cid}")
        
        # 初始化各項計數器
        total_events = len(df)
        skipped_no_dataset = 0
        skipped_too_short = 0
        skipped_inventory_error = 0
        skipped_remove_response_error = 0
        processed_count = 0

        dtfl = h5py.File(hdf5_file, 'r')
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"chunk{cid}"):
            trace_name = row['trace_name']
            dataset_path = f"data/{trace_name}"
            if dataset_path not in dtfl:
                skipped_no_dataset += 1
                continue
            
            dataset = dtfl[dataset_path]
            data = np.array(dataset)
            if data.shape[0] < 6000:
                skipped_too_short += 1
                continue  # 忽略太短的波形
            
            # 建立快取 key 避免重複取得 inventory
            key = (dataset.attrs['network_code'],
                   dataset.attrs['receiver_code'],
                   dataset.attrs['trace_start_time'])
            if key in inventory_cache:
                inventory = inventory_cache[key]
            else:
                try:
                    inventory = client.get_stations(
                        network=dataset.attrs['network_code'],
                        station=dataset.attrs['receiver_code'],
                        starttime=UTCDateTime(dataset.attrs['trace_start_time']),
                        endtime=UTCDateTime(dataset.attrs['trace_start_time']) + 60,
                        loc="*", 
                        channel="*",
                        level="response"
                    )
                    inventory_cache[key] = inventory
                except FDSNNoDataException as e:
                    print(f"⚠️ 無儀器響應資料: {e}，跳過此筆資料")
                    skipped_inventory_error += 1
                    continue
                except Exception as e:
                    print(f"⚠️ 取得 inventory 時發生錯誤: {e}，跳過此筆資料")
                    skipped_inventory_error += 1
                    continue
            
            # 將 dataset 轉成 ObsPy stream，並移除儀器響應（轉成加速度）
            st = make_stream(dataset)
            try:
                st.remove_response(inventory=inventory, output="ACC", plot=False)
            except Exception as e:
                print(f"⚠️ remove_response 發生錯誤: {e}，跳過此筆資料")
                skipped_remove_response_error += 1
                continue
            
            # Trim 每個 trace 至前 6000 點 (結果 shape: (3, 6000))
            processed_data = np.stack([tr.data[:6000] for tr in st], axis=0)
            
            # 儲存部分 metadata
            meta = [
                row['source_distance_km'],
                row['source_depth_km'],
                row['source_magnitude'],
                row['receiver_latitude'],
                row['receiver_longitude'],
                row['source_latitude'],
                row['source_longitude']
            ]
            
            all_waveforms.append(processed_data)
            all_metadata.append(meta)
            processed_count += 1
        
        dtfl.close()
        print(f"📊 Chunk {cid} 統計: 總共 {total_events} 筆, 處理 {processed_count} 筆, "
              f"跳過(無 dataset: {skipped_no_dataset}, 太短: {skipped_too_short}, "
              f"inventory錯誤: {skipped_inventory_error}, remove_response錯誤: {skipped_remove_response_error})")
    
    if not all_waveforms:
        raise ValueError("沒有任何資料被處理，請檢查輸入條件或儀器響應設定。")
    
    waveforms = np.stack(all_waveforms, axis=0)        # shape: (N, 3, 6000)
    metadata = np.array(all_metadata, dtype=np.float32)  # shape: (N, 7)
    return waveforms, metadata

# -------------------------------
# 主程式：處理波形並儲存成加速度資料
# -------------------------------
def main():
    base_path = r"Z:\seismic\STEAD"  # 請調整成你實際的資料夾路徑
    chunk_ids = [2, 3, 4, 5, 6]
    
    # 取得加速度波形與 metadata
    waveforms, metadata = process_chunks(chunk_ids, base_path)
    output_file = os.path.join(base_path, "stead_acceleration_chunks.npz")
    np.savez(output_file, waveforms=waveforms, metadata=metadata)
    print(f"\n✅ Saved {waveforms.shape[0]} acceleration waveforms to {output_file}")

if __name__ == '__main__':
    main()
