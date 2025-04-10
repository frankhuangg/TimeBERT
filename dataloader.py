# dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class SeismicDataset(Dataset):
    def __init__(self, waveform_list, metadata, patch_size=100, mask_ratio=0.25):
        """
        waveform_list: 波形資料列表，形狀 (N, 3, T)
        metadata: 其他元資料，形狀 (N, 7)，其中第 4、5 欄為經緯度資訊
        patch_size: 分割波形片段的大小
        mask_ratio: 遮蔽比例，用於 MPM
        """
        self.waveforms = waveform_list
        self.metadata = metadata
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # 根據 metadata 中的經緯度組合站點識別字串，例如 "lat_lon"
        station_coords = [f"{lat:.5f}_{lon:.5f}" for lat, lon in self.metadata[:, 3:5]]
        self.label_encoder = LabelEncoder()
        self.domain_ids = self.label_encoder.fit_transform(station_coords)

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        # 取得單筆波形資料並轉成 tensor
        waveform = self.waveforms[idx].copy()  # (3, T)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        # 標準化處理（沿時間軸計算各 channel 的均值與標準差）
        mean = waveform.mean(dim=1, keepdim=True)
        std = waveform.std(dim=1, keepdim=True) + 1e-5
        waveform = (waveform - mean) / std

        domain_id = self.domain_ids[idx]
        C, T = waveform.shape

        # VAR 任務：對波形做變量替換處理
        waveform_for_var = waveform.clone()
        var_mask = torch.zeros(3)
        replace_idx = torch.randint(0, 3, (1,)).item()
        other_idx = (idx + 1) % len(self.waveforms)
        waveform_for_var[replace_idx] = torch.tensor(self.waveforms[other_idx][replace_idx], dtype=torch.float32)
        var_mask[replace_idx] = 1
        # 重新標準化被替換的 channel
        channel_data = waveform_for_var[replace_idx]
        channel_mean = channel_data.mean()
        channel_std = channel_data.std() + 1e-5
        waveform_for_var[replace_idx] = (channel_data - channel_mean) / channel_std

        num_patches = T // self.patch_size
        mpm_mask = torch.rand(C, num_patches) < self.mask_ratio

        return {
            'waveform': waveform,          # 原始輸入資料（用於 MPM/DOM 任務）
            'waveform_var': waveform_for_var,  # VAR 任務資料
            'domain_id': domain_id,
            'var_mask': var_mask,
            'mpm_mask': mpm_mask
        }

def prepare_dataloader_from_npz(npz_path, batch_size=32, train_ratio=0.7, val_ratio=0.1, seed=42):
    """
    npz_path: 儲存 waveforms 與 metadata 的 npz 檔案路徑
    train_ratio: 訓練集比例
    val_ratio: 驗證集比例，測試集比例即為 1 - (train_ratio + val_ratio)
    """
    data = np.load(npz_path)
    waveforms = data['waveforms']
    metadata = data['metadata']

    # 先拆分出測試集
    test_size = 1 - (train_ratio + val_ratio)
    X_train_val, X_test, meta_train_val, meta_test = train_test_split(
        waveforms, metadata, test_size=test_size, random_state=seed, shuffle=True
    )
    # 從剩下資料中拆分出驗證集
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, meta_train, meta_val = train_test_split(
        X_train_val, meta_train_val, test_size=val_ratio_adjusted, random_state=seed, shuffle=True
    )

    train_set = SeismicDataset(X_train, meta_train)
    val_set = SeismicDataset(X_val, meta_val)
    test_set = SeismicDataset(X_test, meta_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
