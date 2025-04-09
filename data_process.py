import os
import h5py
import pandas as pd
import numpy as np

def preprocess_data(base_dirs):
    processed_waveforms = []
    event_metadata = []
    
    for base_dir in base_dirs:
        for file in os.listdir(base_dir):
            if file.startswith("metadata_") and file.endswith(".csv"):
                metadata_path = os.path.join(base_dir, file)
                print("Processing metadata file:", metadata_path)
                df = pd.read_csv(metadata_path)
                
                # 篩選條件
                df = df[df["station_code"] == "HWA"]
                df = df[df["trace_snr_db"] >= 10.0]
                df = df[df["trace_p_arrival_sample"] >= 5 * 100]
                df = df[df["station_location_code"] == 10]
                
                if df.empty:
                    print("No records meeting filtering criteria in", metadata_path)
                    continue
                
                # 從 metadata 檔名取得年份
                year = file.split("_")[1].split(".")[0]
                possible_names = [f"chunks_{year}.h5", f"chunks_{year}.hdf5", f"chunks_{year}"]
                hdf5_file_path = None
                for name in possible_names:
                    candidate = os.path.join(base_dir, name)
                    if os.path.exists(candidate):
                        hdf5_file_path = candidate
                        break
                if hdf5_file_path is None:
                    print(f"No hdf5 file found for year {year} in {base_dir}")
                    continue
                
                print("Using hdf5 file:", hdf5_file_path)
                
                with h5py.File(hdf5_file_path, "r") as h5f:
                    data_group = h5f["data"]
                    
                    for idx, row in df.iterrows():
                        trace_name = row["trace_name"]
                        if trace_name not in data_group:
                            print(f"Trace {trace_name} not found in {hdf5_file_path}")
                            continue
                        
                        waveform = data_group[trace_name][()]
                        
                        # 提取 window
                        p_arrival_sample = row["trace_p_arrival_sample"]
                        start_idx = max(0, int(p_arrival_sample) - 500)
                        end_idx = min(waveform.shape[1], int(p_arrival_sample) + 2500)
                        waveform_window = np.pad(
                            waveform[:, start_idx:end_idx], 
                            ((0, 0), (max(0, 500 - int(p_arrival_sample)), max(0, (int(p_arrival_sample) + 2500) - waveform.shape[1]))), 
                            mode='constant'
                        )
                        
                        # Min-Max Normalization
                        waveform_window = (waveform_window - waveform_window.min()) / (waveform_window.max() - waveform_window.min() + 1e-8)
                        
                        # 儲存事件資訊
                        event_info = [
                            row["path_ep_distance_km"],
                            row["station_latitude_deg"],
                            row["station_longitude_deg"],
                            row["source_magnitude"],
                            row["source_depth_km"],
                            row["source_latitude_deg"],
                            row["source_longitude_deg"]
                        ]
                        
                        processed_waveforms.append(waveform_window.astype(np.float32))
                        event_metadata.append(event_info)
    
    return processed_waveforms, event_metadata

def main():
    base_dirs = ["Y:\\CWB24\\CWASN\\2012_2014", "Y:\\CWB24\\CWASN\\2015_2018", "Y:\\CWB24\\CWASN\\2019_2021"]
    
    waveforms, metadata = preprocess_data(base_dirs)
    
    if waveforms:
        waveforms = np.stack(waveforms, axis=0)  # shape: (N, 3, 3000)
        metadata = np.array(metadata, dtype=np.float32)  # shape: (N, 7)
        
        np.savez("preprocessed_data_with_metadata.npz", waveforms=waveforms, metadata=metadata)
        print("Saved preprocessed data with metadata to preprocessed_data_with_metadata.npz")

if __name__ == "__main__":
    main()
