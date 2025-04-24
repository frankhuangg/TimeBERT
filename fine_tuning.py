import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from CWA_dataloader import create_dataloaders_7_2_1, get_num_domains
from TimeBERT_pretrain import TimesBERTForSeismic, SeismicDataset
import matplotlib.pyplot as plt


def train_model(ft_model, train_loader, val_loader, device, epochs=100, lr=1e-4):
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
            torch.save(ft_model.state_dict(), "fine_tuned_timesbert_pga_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"⏹️ Early stopping at epoch {epoch} (no improvement for {early_stop_patience} epochs)")
                break

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
    plt.savefig('fine_tune_epoch_loss.png')
    print("✅ Saved loss plot to fine_tune_epoch_loss.png")

# --- 2. 定義微調模型 ---
class TimesBERTForPGA(nn.Module):
    def __init__(self, pretrained: TimesBERTForSeismic):
        super().__init__()
        # 保留原預訓練模型的 embedding 和 encoder
        self.embedding = pretrained.embedding
        self.encoder = pretrained.encoder
        # 回歸頭：從 [DOM] token 特徵預測單一標量
        D = pretrained.embedding.embed_dim
        self.regressor = nn.Linear(D, 1)
        # self.regressor = nn.Sequential(
        #     nn.Linear(D, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, 1)
        # )

    def forward(self, waveform, domain_id):
        # 1. Patch embedding
        tokens, _, dom_tokens, _ = self.embedding(waveform, domain_id)
        B, C, num_patches, D = tokens.shape
        # 2. 拼接 DOM token 與 patch tokens
        patch_seq = tokens.view(B, C * num_patches, D)
        all_tokens = torch.cat([dom_tokens.unsqueeze(1), patch_seq], dim=1)
        # 3. Transformer encoder
        encoded = self.encoder(all_tokens.transpose(0,1)).transpose(0,1)  # (B, 1 + C*num_patches, D)
        # 4. 取 [DOM] token 作為聚合特徵
        feat = encoded[:, 0, :]  # (B, D)
        # 5. 線性回歸
        pga_pred = self.regressor(feat).squeeze(-1)  # (B,)
        return pga_pred

# --- 3. 訓練與驗證函式 ---
def train_pga(model, loader, optimizer, device):
    model.train()
    mae = nn.L1Loss()
    total_loss = 0.0
    for batch in loader:
        x = batch['waveform'].to(device)
        d = batch['domain_id'].to(device)
        y = batch['pga'].to(device)
        optimizer.zero_grad()
        y_hat = model(x, d)
        loss = mae(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_pga(model, loader, device):
    model.eval()
    mae = nn.L1Loss()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch['waveform'].to(device)
            d = batch['domain_id'].to(device)
            y = batch['pga'].to(device)
            y_hat = model(x, d)
            total_loss += mae(y_hat, y).item()

    return total_loss / len(loader)


if __name__ == '__main__':
    # 讀取資料並切分
    metadata_csv = "CWA_processed_data/all_metadata.csv"
    hdf5_path = "CWA_processed_data/all.hdf5"
    (train_wave, train_meta), (test_wave, test_meta), (val_wave, val_meta) = \
        create_dataloaders_7_2_1(hdf5_path, metadata_csv)

    # 建立回歸用 Dataset & DataLoader
    batch_size = 32

    train_ds = SeismicDataset(train_wave, train_meta)
    val_ds = SeismicDataset(val_wave, val_meta)
    test_ds = SeismicDataset(test_wave, test_meta)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader   = DataLoader(test_ds,   batch_size=batch_size, shuffle=False)

    # 載入預訓練模型權重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_domains = get_num_domains(metadata_csv)
    pretrained = TimesBERTForSeismic(patch_size=100, embed_dim=768, num_domains=num_domains)
    pretrained.load_state_dict(torch.load("pretrained_timesbert.pt", map_location=device))
    pretrained.to(device)

    # 建立微調模型
    ft_model = TimesBERTForPGA(pretrained).to(device)

    # 執行訓練
    train_model(ft_model, train_loader, val_loader, device, epochs=100, lr=1e-4)

    test_loss = evaluate_pga(ft_model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")
    # 儲存微調後模型
    torch.save(ft_model.state_dict(), "fine_tuned_timesbert_pga.pt")