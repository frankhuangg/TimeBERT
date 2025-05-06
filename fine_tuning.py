import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from CWA_dataloader import create_dataloaders_7_2_1, get_num_domains
from TimeBERT_pretrain import TimesBERTForSeismic, SeismicDataset
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)

    def forward(self, x):  # x: (B, T, D)
        weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return (weights * x).sum(dim=1)  # (B, D)
    
def train_model(ft_model, train_loader, val_loader, device, epochs=100, lr=1e-3):
    ft_model = ft_model.to(device)
    optimizer = torch.optim.AdamW(ft_model.parameters(), lr=lr)
    ft_model.train()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10


    # 用於記錄各個 epoch 的損失
    train_losses = []
    val_losses = []
    for epoch in range(1, epochs+1):
        train_loss = train_pga(ft_model, train_loader, optimizer, device)
        val_loss   = evaluate_pga(ft_model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        print(f"Epoch {epoch}: Train MAE={train_loss:.4f}, Val MAE={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ft_model.state_dict(), "fine_tuned_timesbert_pga.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"⏹️ Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                break
    print(f"best MAE={best_val_loss:.4f}")

    # 繪製 epoch loss 曲線
    epochs_list = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(10,6))
    plt.plot(epochs_list, train_losses, marker='o', label='Train MAE')
    plt.plot(epochs_list, val_losses,   marker='x', label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE Loss')
    plt.title('Fine-tuning PGA Regression Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('fine_tune_epoch_loss_after3s.png')
    print("✅ Saved loss plot to fine_tune_epoch_loss_after3s.png")

# --- 2. 定義微調模型 ---
class TimesBERTForPGA(nn.Module):
    def __init__(self, pretrained: TimesBERTForSeismic):
        super().__init__()
        # 保留原預訓練模型的 embedding 和 encoder
        self.embedding = pretrained.embedding
        self.encoder = pretrained.encoder
        # 回歸頭：從 [DOM] token 特徵預測單一標量
        D = pretrained.embedding.embed_dim
        self.attn_pool = AttentionPooling(D)
        self.regressor = nn.Linear(D, 1)
        # self.regressor = nn.Sequential(
        #     nn.Linear(D, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, 1)
        # )

    def forward(self, waveform, domain_id):
        # 1. Patch embedding
        tokens, var_tokens, dom_tokens, patches = self.embedding(waveform, domain_id)
        B, C, num_patches, D = tokens.shape
        # 動態處理：只取對應長度的 position embedding
        tokens = tokens + self.embedding.pos_encoder.pe[:, :num_patches, :].unsqueeze(1)
        # 2. 拼接 DOM token 與 patch tokens
        patch_seq = tokens.view(B, C * num_patches, D)
        all_tokens = torch.cat([dom_tokens.unsqueeze(1), patch_seq], dim=1)
        # 3. Transformer encoder
        encoded = self.encoder(all_tokens.transpose(0,1)).transpose(0,1)  # (B, 1 + C*num_patches, D)
        # 4. 取 [DOM] token 作為聚合特徵
        # feat = encoded[:, 0, :]  # (B, D)
        feat = self.attn_pool(encoded)  # (B, D)

        # 5. 線性回歸
        pga_pred = self.regressor(feat).squeeze(-1)  # (B,)
        return pga_pred

# def crop_pwave_window(waveform_list, metadata_list, pre_sec=3, post_sec=3, sample_rate=100):
#     cropped_waveforms = []
#     for waveform, meta in zip(waveform_list, metadata_list):
#         p_arrival_sec = meta['p_arrival_sample'] / sample_rate  # 取得 P 波到時的秒數
#         p_sample = int(p_arrival_sec * sample_rate)
#         pre_samples = int(pre_sec * sample_rate)
#         post_samples = int(post_sec * sample_rate)
        
#         start_idx = max(p_sample - pre_samples, 0)
#         end_idx = min(p_sample + post_samples, waveform.shape[-1])

#         cropped_wave = waveform[:, start_idx:end_idx]  # (3, new_T)
#         cropped_waveforms.append(cropped_wave)
#     return cropped_waveforms


# --- 3. 訓練與驗證函式 ---
def train_pga(model, loader, optimizer, device):
    model.train()
    mae = nn.L1Loss()
    total_weighted_loss = 0.0
    total_weight = 0.0
    total_loss = 0.0
    for batch in loader:
        x = batch['waveform'].to(device)
        d = batch['domain_id'].to(device)
        y = batch['pga'].to(device)
        y_hat = model(x, d)

        # 根據 PGA 加權
        weights = torch.tensor([pga_to_weight(v.item()) for v in y], device=device, dtype=torch.float32)
        loss = mae(y_hat, y)
        weighted_loss = (loss * weights).sum()
        weight_sum = weights.sum() 

        # loss.backward()
        optimizer.zero_grad()
        loss.backward()
        # (weighted_loss / weight_sum).backward()
        optimizer.step()

        total_loss += loss.item()
        total_weighted_loss += weighted_loss.item()
        total_weight += weight_sum.item()
    # return total_weighted_loss / total_weight if total_weight > 0 else 0.0
    return total_loss / len(loader)


def evaluate_pga(model, loader, device):
    model.eval()
    mae = nn.L1Loss()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x = batch['waveform'].to(device)
            d = batch['domain_id'].to(device)
            y = batch['pga'].to(device)
            y_hat = model(x, d)
            total_loss += mae(y_hat, y).item()
            all_preds.append(y_hat.detach().cpu())
            all_targets.append(y.detach().cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    # 加上 log10（避免 log(0) 錯誤，加上 epsilon）
    eps = 1e-5
    # log_preds = np.log10(all_preds + eps)
    # log_targets = np.log10(all_targets + eps)
    # log_errors = log_preds - log_targets
    errors = all_preds - all_targets
    mae = np.mean(np.abs(errors))
    std = np.std(errors)
    # 直方圖（誤差）
    fig, ax = plt.subplots(figsize=(8, 5))
    # 直接用固定範圍與 bin
    bins = np.arange(-20, 21, 1)   # –20 到 20，每 1 單位一格
    ax.hist(errors, bins=bins, edgecolor='black', color='salmon', alpha=0.6)
    # 設定 X 軸只顯示這幾個刻度
    xticks = [-20, -10, 0, 10, 20]
    ax.set_xticks(xticks)
    ax.set_xlim(xticks[0], xticks[-1])
    # 避免科學記號（可選，如果數值小就不會出現）
    ax.xaxis.set_major_formatter(ScalarFormatter())
    # 調整刻度標籤字型與方向
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    # 5. 標題與座標軸
    ax.set_title(f"PGA Error\nMAE = {mae:.3f}, STD = {std:.2f}", fontsize=14)
    ax.set_xlabel("PGA (cm/s²)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)

    # 6. 加 Y 軸格線
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("pga_prediction_error_hist.png", dpi=150)
    plt.close()
    # 畫散點圖
    plt.figure(figsize=(6, 6))
    plt.scatter(all_targets, all_preds, alpha=0.4, edgecolors='k', linewidths=0.5)
    min_val = 0
    max_val = max(all_targets.max(), all_preds.max())
    plt.plot([min_val, max_val],
         [min_val, max_val],
         'r--', lw=2)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)  
    plt.xlabel("Ground Truth PGA")
    plt.ylabel("Predicted PGA")
    plt.title(f"PGA Prediction Scatter\nMAE={mae:.4f}, STD={std:.4f}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.savefig("pga_prediction_scatter.png", dpi=150)
    plt.close()

    return total_loss / len(loader)

def pga_to_weight(pga):
    if pga < 0.8:
        return 0  # 小於 0.8 的資料會被忽略
    elif pga < 2.5:
        return 1
    elif pga < 8.0:
        return 2
    elif pga < 25:
        return 3
    elif pga < 80:
        return 4
    else:
        return 5


if __name__ == '__main__':
    # 讀取資料並切分
    local_dir = r"local_output"
    batch_size = 32
    fs = 100  # 取樣率
    start = (5 - 3) * fs   # P 點前 3s 在原始區段中的起始索引
    end   = (5 + 3) * fs   # P 點後 3s 在原始區段中的結束索引
    patch_size = 100

    # Train split
    train_wave = np.load(os.path.join(local_dir, 'train_wave_balanced_20k.npy'))
    train_meta = pd.read_csv(os.path.join(local_dir, 'train_meta_balanced_20k.csv'))
    # train_wave_cropped = crop_pwave_window(train_wave, train_meta)
    train_wave = train_wave[:, :, start:end]
    train_dataset = SeismicDataset(train_wave, train_meta, patch_size=patch_size)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    # Valid split
    val_wave = np.load(os.path.join(local_dir, 'val_wave.npy'))
    val_meta = pd.read_csv(os.path.join(local_dir, 'val_meta.csv'))
    # val_wave_cropped = crop_pwave_window(val_wave, val_meta)
    val_wave = val_wave[:, :, start:end]
    val_dataset = SeismicDataset(val_wave, val_meta, patch_size=patch_size)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    # Test split
    test_wave = np.load(os.path.join(local_dir, 'test_wave.npy'))
    test_meta = pd.read_csv(os.path.join(local_dir, 'test_meta.csv'))
    # test_wave_cropped = crop_pwave_window(test_wave, test_meta)
    test_wave = test_wave[:, :, start:end]
    test_dataset = SeismicDataset(test_wave, test_meta, patch_size=patch_size)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    
    # 計算總 domain 數
    num_domains = (
        get_num_domains(os.path.join(local_dir, 'train_meta_balanced_20k.csv'))
        + get_num_domains(os.path.join(local_dir, 'val_meta.csv'))
        + get_num_domains(os.path.join(local_dir, 'test_meta.csv'))
    )

    # 載入預訓練模型權重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained = TimesBERTForSeismic(patch_size=patch_size, embed_dim=768, num_domains=num_domains)
    pretrained.load_state_dict(torch.load("pretrained_timesbert_patch100.pt", map_location=device))
    pretrained.to(device)

    # 建立微調模型
    ft_model = TimesBERTForPGA(pretrained).to(device)

    # 執行訓練
    train_model(ft_model, train_loader, val_loader, device, epochs=100, lr=1e-3)

    test_loss = evaluate_pga(ft_model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")
    # 儲存微調後模型
    # torch.save(ft_model.state_dict(), "fine_tuned_timesbert_pga.pt")
