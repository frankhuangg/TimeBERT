import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from CWA_dataloader import cwa_loader, SELECTED_CWA_ATTRS, collate_fn_skip_none

# ------------------------------
# 1. SeismicDataset - 地震波形資料集（包含站點的經緯度資訊轉換成 Domain ID）
# ------------------------------
class SeismicDataset(Dataset):
    def __init__(self, waveform_list, metadata, patch_size = 100, mask_ratio = 0.25):
        """
        waveform_list: 波形資料列表，形狀為 (N, 3, T) 代表 N 筆資料，每筆有 3 個變數、T 個時間點
        metadata: 每筆資料的其他元資料，形狀為 (N, 7)，其中第 4、5 欄為經緯度
        patch_size: 分割波形片段的大小
        mask_ratio: 在 MPM (Masked Patch Modeling) 中遮蔽片段的比例
        """
        self.waveforms = waveform_list  # 原始波形資料
        self.metadata = metadata        # 對應的元資料（包含經緯度資訊）
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # 根據經緯度資訊，組合成站點識別字串，例如 "lat_lon"
        station_coords = [f"{lat:.5f}_{lon:.5f}" for lat, lon in self.metadata[:, 3:5]]
        # 使用 LabelEncoder 將站點識別字串轉換成數值型的 domain_id
        self.label_encoder = LabelEncoder()
        self.domain_ids = self.label_encoder.fit_transform(station_coords)

        # 可以檢查生成的 domain_id 範圍是否符合要求（此處檢查最大值是否小於 100）
        # print("Domain id range:", self.domain_ids.min(), self.domain_ids.max())
        # assert self.domain_ids.max() < 100, "Domain id exceeds the maximum allowed index for embedding."

    def __len__(self):
        # 返回資料筆數
        return len(self.waveforms)

    def __getitem__(self, idx):
        # 取得指定索引的波形資料與相關資訊
        waveform = self.waveforms[idx].copy()  # 取得一筆波形資料，形狀 (3, T)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        # --- 加入標準化處理 --- #
        # 針對每個 channel 計算均值與標準差（沿時間軸 T）
        mean = waveform.mean(dim=1, keepdim=True)
        std = waveform.std(dim=1, keepdim=True) + 1e-5  # 加入 epsilon 防止除以 0
        waveform = (waveform - mean) / std
        # ------------------------ #
        domain_id = self.domain_ids[idx]         # 對應的 domain_id
        C, T = waveform.shape

        # 替換的變量會影響到其他任務

        # # 模擬變量替換（Variate Replacement）：隨機選擇一個變量，將其用另一筆資料相同變量的波形來替換
        # var_mask = torch.zeros(C)  # 建立變量替換遮罩，初始為 0 (無替換)
        # replace_idx = torch.randint(0, C, (1,)).item()  # 隨機選擇一個變量索引進行替換
        # # 這裡選擇下一筆資料作為替換來源（循環使用資料）
        # other_idx = (idx + 1) % len(self.waveforms)
        # waveform[replace_idx] = torch.tensor(self.waveforms[other_idx][replace_idx], dtype=torch.float32)
        # var_mask[replace_idx] = 1  # 標記被替換的變量

         # 給 VAR 任務用的副本（可以被污染）
        waveform_for_var = waveform.clone()
    
        # 通道替換僅作用於這一份 waveform_for_var
        var_mask = torch.zeros(3)
        replace_idx = torch.randint(0, 3, (1,)).item()
        other_idx = (idx + 1) % len(self.waveforms)
        waveform_for_var[replace_idx] = torch.tensor(self.waveforms[other_idx][replace_idx], dtype=torch.float32)
        var_mask[replace_idx] = 1

        # 假設替換完後，只重新標準化被替換的 channel
        channel_data = waveform_for_var[replace_idx]
        channel_mean = channel_data.mean()
        channel_std = channel_data.std() + 1e-5
        waveform_for_var[replace_idx] = (channel_data - channel_mean) / channel_std

        num_patches = T // self.patch_size
        mpm_mask = torch.rand(C, num_patches) < self.mask_ratio

        return {
            'waveform': waveform,          # 原始輸入 → MPM / DOM 用
            'waveform_var': waveform_for_var,     # 替換後 → VAR 任務專用
            'domain_id': self.domain_ids[idx],
            'var_mask': var_mask,
            'mpm_mask': mpm_mask
        }

# ------------------------------
# 2. PatchEmbedding - 將波形切成片段並做線性投影，同時加入特殊的變量與站點嵌入
# ------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=100, embed_dim=768):
        """
        patch_size: 每個 patch 的時間點數量
        embed_dim: 嵌入向量的維度
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # 使用線性層將每個 patch 投影到嵌入空間
        self.linear = nn.Linear(patch_size, embed_dim)
        # 為每個變量（總共 3 個）建立專屬的嵌入向量
        self.var_embed = nn.Embedding(3, embed_dim)
        # 為每個站點（domain）建立嵌入，最大站點數設定為 1000
        self.dom_embed = nn.Embedding(1000, embed_dim)  # max 1000 stations

    def forward(self, waveform, domain_ids):
        """
        waveform: 輸入波形資料，形狀 (B, C, T)
        domain_ids: 每個序列對應的站點 id，形狀 (B,)
        """
        B, C, T = waveform.shape
        num_patches = T // self.patch_size
        # 只使用完整的 patch 部分，並 reshape 成 (B, C, num_patches, patch_size)
        patches = waveform[:, :, :num_patches * self.patch_size].reshape(B, C, num_patches, self.patch_size)
        # 線性投影得到 tokens，形狀 (B, C, num_patches, embed_dim)
        tokens = self.linear(patches)

        # 為每個變量加入專屬的 [VAR] token
        var_tokens = self.var_embed(torch.arange(C, device=waveform.device))  # (C, embed_dim)
        var_tokens = var_tokens.unsqueeze(0).expand(B, -1, -1)  # 擴展成 (B, C, embed_dim)

        # 為每個序列加入站點 [DOM] token
        dom_tokens = self.dom_embed(domain_ids)  # (B, embed_dim)
        return tokens, var_tokens, dom_tokens, patches

# ------------------------------
# 3. TimesBERTEncoder - Transformer 編碼器作為骨幹網路
# ------------------------------
class TimesBERTEncoder(nn.Module):
    def __init__(self, embed_dim = 768, num_layers = 6):
        """
        embed_dim: 輸入 token 的嵌入維度
        num_layers: Transformer 層數
        """
        super().__init__()
        # 建立單層 Transformer 編碼器層
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead = 8, dim_feedforward = 2048)
        # 堆疊多個 Transformer 編碼器層
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

    def forward(self, x):
        """
        x: 輸入 token 序列，形狀需要符合 Transformer 的要求
        """
        return self.encoder(x)

# ------------------------------
# 4. FTPHead - 功能性 Token 預測頭（用於變量與站點分類）
# ------------------------------
class FTPHead(nn.Module):
    def __init__(self, embed_dim, num_domains):
        """
        embed_dim: 輸入 token 的嵌入維度
        num_domains: 站點總數（domain 的數量）
        """
        super().__init__()
        # 用於變量分類，輸出 2 個類別（例如原始與替換）
        self.var_classifier = nn.Linear(embed_dim, 2)
        # 用於站點分類
        self.dom_classifier = nn.Linear(embed_dim, num_domains)

    def forward(self, z_var, z_dom, var_labels, dom_labels):
        """
        z_var: 來自最後一層 Transformer 的變量 token 特徵
        z_dom: 來自最後一層 Transformer 的站點 token 特徵
        var_labels: 真實的變量標籤 (是否替換)
        dom_labels: 真實的站點 id
        """
        # 對變量 token 進行分類，並計算交叉熵損失
        loss_var = F.cross_entropy(self.var_classifier(z_var.reshape(-1, z_var.size(-1))), var_labels.reshape(-1).long())
        # 對站點 token 進行分類，並計算交叉熵損失
        loss_dom = F.cross_entropy(self.dom_classifier(z_dom), dom_labels.long())
        # 返回總損失以及各自的損失分量
        return loss_var + loss_dom, loss_var, loss_dom

# ------------------------------
# 5. TimesBERTForSeismic - 整合模型，包含嵌入、Transformer 編碼器、FTP 預測頭以及 MPM 預測頭
# ------------------------------
class TimesBERTForSeismic(nn.Module):
    def __init__(self, patch_size = 100, embed_dim = 768, num_domains = 1000):
        """
        patch_size: 每個 patch 的大小
        embed_dim: token 嵌入的維度
        num_domains: 站點（domain）的數量
        """
        super().__init__()
        self.patch_size = patch_size
        # 初始化 patch 嵌入模塊
        self.embedding = PatchEmbedding(patch_size, embed_dim)
        # 初始化 Transformer 編碼器
        self.encoder = TimesBERTEncoder(embed_dim)
        # 初始化 FTP 預測頭（功能性 token 預測頭）
        self.ftp = FTPHead(embed_dim, num_domains)
        # 為 MPM (Masked Patch Modeling) 建立投影頭，將編碼器輸出轉回 patch 原始維度
        self.projection_head = nn.Linear(embed_dim, patch_size)

    def forward(self, waveform, waveform_var, domain_ids, var_mask, mpm_mask):
        """
        waveform: 輸入波形，形狀 (B, C, T)
        domain_ids: 每筆資料對應的站點 id，形狀 (B,)
        var_mask: 變量替換遮罩，形狀 (B, C)
        mpm_mask: MPM 遮罩，形狀 (B, C, num_patches)
        """
        B, C, T = waveform.shape
        # 將波形進行 patch 嵌入，並獲得變量與站點特殊 token，以及原始 patch 資料
        tokens, var_tokens, dom_tokens, original_patches = self.embedding(waveform, domain_ids)

        # 複製 tokens 用於遮蔽處理（MPM）：將被遮蔽的 tokens 設為 0
        tokens_masked = tokens.clone()
        # 擴展 mpm_mask 使其形狀與 tokens_masked 相同
        mask_expanded = mpm_mask.unsqueeze(-1).expand_as(tokens_masked)
        tokens_masked[mask_expanded] = 0

        # 將所有 tokens 合併：先加入 [DOM] token，再加入所有 patch tokens，最後加入 [VAR] tokens
        all_tokens = torch.cat([
            dom_tokens.unsqueeze(1),                      # (B, 1, embed_dim)
            tokens_masked.view(B, -1, tokens.size(-1)),     # (B, C*num_patches, embed_dim)
            var_tokens                                      # (B, C, embed_dim)
        ], dim=1)
        

        # 將 token 序列轉置成 Transformer 所需的形狀 (sequence_length, B, embed_dim)
        encoded = self.encoder(all_tokens.transpose(0, 1)).transpose(0, 1)
        # 分離出 [DOM] token 的編碼結果
        z_dom = encoded[:, 0, :]
        # 替代的 waveform_var → 重新 embedding → 拿 VAR token
        _, var_tokens_alt, _, _ = self.embedding(waveform_var, domain_ids)
        all_tokens_var = torch.cat([
            dom_tokens.unsqueeze(1),  # 同樣站台
            tokens.view(waveform.size(0), -1, tokens.size(-1)),
            var_tokens_alt
        ], dim=1)
        encoded_var = self.encoder(all_tokens_var.transpose(0, 1)).transpose(0, 1)
        
        # 分離出 [VAR] tokens 的編碼結果
        z_var = encoded_var[:, -C:, :]

        # 取出 patch tokens 的部分，並重構回 (B, C, num_patches, embed_dim)
        patch_token_start = 1
        patch_token_end = 1 + C * tokens.size(2)
        encoded_patches = encoded[:, patch_token_start:patch_token_end, :].view(B, C, -1, encoded.size(-1))
        # 透過投影頭將 patch token 的特徵映射回原始 patch 的大小
        pred_patches = self.projection_head(encoded_patches)
        # 如果有任何 patch 被遮蔽，計算 MPM 的均方誤差損失
        if mpm_mask.sum() > 0:
            mpm_loss = F.mse_loss(pred_patches[mpm_mask], original_patches[mpm_mask])
        else:
            mpm_loss = 0.0

        # 計算 FTP 的損失（包含變量分類與站點分類）
        ftp_loss, loss_var, loss_dom = self.ftp(z_var, z_dom, var_mask, domain_ids)

        # 返回總損失及各部分損失
        return mpm_loss + ftp_loss, mpm_loss, loss_var, loss_dom

# ------------------------------
# 6. train_model - 模型訓練循環
# ------------------------------
def train_model(model, train_loader, val_loader, device, num_epochs = 100, lr = 1e-4):
    """
    model: 待訓練的 TimesBERTForSeismic 模型
    dataloader: 提供訓練資料的 DataLoader
    device: 執行訓練的設備（如 'cuda' 或 'cpu'）
    num_epochs: 訓練的總 epoch 數
    lr: 優化器的學習率
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)

    model.train()  # 設置模型為訓練模式

    # 用來儲存每個 epoch 的各項 loss
    epoch_total_loss = []
    epoch_mpm_loss = []
    epoch_var_loss = []
    epoch_dom_loss = []
    val_total_loss = []
    val_mpm_loss = []
    val_var_loss = []
    val_dom_loss = []

    for epoch in range(num_epochs):
        total_loss = 0
        total_mpm_loss = 0
        total_var_loss = 0
        total_dom_loss = 0

        # 逐批次讀取資料
        for batch in tqdm(train_loader, desc = f"Epoch {epoch+1}"):
            # 將批次資料轉移到指定設備上
            waveform = batch['waveform'].to(device)
            domain_id = batch['domain_id'].to(device)
            var_mask = batch['var_mask'].to(device)
            mpm_mask = batch['mpm_mask'].to(device)
            waveform_var = batch['waveform_var'].to(device)

            optimizer.zero_grad()  # 梯度歸零
            # 前向傳播並獲取損失值
            loss, mpm_loss, var_loss, dom_loss = model(waveform, waveform_var, domain_id, var_mask, mpm_mask)
            loss.backward()  # 反向傳播
            optimizer.step()  # 優化器更新參數

            # 累積各項損失
            total_loss += loss.item()
            total_mpm_loss += mpm_loss.item()
            total_var_loss += var_loss.item()
            total_dom_loss += dom_loss.item()

        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_mpm_loss = total_mpm_loss / num_batches
        avg_var_loss = total_var_loss / num_batches
        avg_dom_loss = total_dom_loss / num_batches
        # # 檢查是否為 tensor，若是則轉移到 CPU 並取出標量
        # if isinstance(avg_loss, torch.Tensor):
        #     avg_loss = avg_loss.cpu().item()
        # if isinstance(avg_mpm_loss, torch.Tensor):
        #     avg_mpm_loss = avg_mpm_loss.cpu().item()
        # if isinstance(avg_var_loss, torch.Tensor):
        #     avg_var_loss = avg_var_loss.cpu().item()
        # if isinstance(avg_dom_loss, torch.Tensor):
        #     avg_dom_loss = avg_dom_loss.cpu().item()
        
        epoch_total_loss.append(avg_loss)
        epoch_mpm_loss.append(avg_mpm_loss)
        epoch_var_loss.append(avg_var_loss)
        epoch_dom_loss.append(avg_dom_loss)
        # 每個 epoch 結束後打印平均損失
        print(f"✅ Epoch {epoch+1} - Avg Total Loss: {avg_loss:.4f} | MPM: {avg_mpm_loss:.4f}, VAR: {avg_var_loss:.4f}, DOM: {avg_dom_loss:.4f}")

        # 📍 Evaluate on validation set
        val_loss, val_mpm, val_var, val_dom = evaluate_model(model, val_loader, device)
        print(f"🔍 Val   - Avg Loss: {val_loss:.4f} | MPM: {val_mpm:.4f}, VAR: {val_var:.4f}, DOM: {val_dom:.4f}")

        # 收集測試 loss
        val_total_loss.append(val_loss)
        val_mpm_loss.append(val_mpm)
        val_var_loss.append(val_var)
        val_dom_loss.append(val_dom)


    # 畫出訓練與驗證 loss 的曲線圖
    epochs = range(1, num_epochs+1)
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, epoch_total_loss, marker='o', label='Train Total Loss')
    plt.plot(epochs, val_total_loss, marker='x', label='Val Total Loss')
    plt.plot(epochs, epoch_mpm_loss, marker='o', label='Train MPM Loss')
    plt.plot(epochs, val_mpm_loss, marker='x', label='Val MPM Loss')
    plt.plot(epochs, epoch_var_loss, marker='o', label='Train VAR Loss')
    plt.plot(epochs, val_var_loss, marker='x', label='Val VAR Loss')
    plt.plot(epochs, epoch_dom_loss, marker='o', label='Train DOM Loss')
    plt.plot(epochs, val_dom_loss, marker='x', label='Val DOM Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_model(model, dataloader, device):
    model.eval()
    model = model.to(device)

    total_loss = 0
    total_mpm = 0
    total_var = 0
    total_dom = 0

    for batch in dataloader:
        waveform = batch['waveform'].to(device)
        waveform_var = batch['waveform_var'].to(device)
        domain_id = batch['domain_id'].to(device)
        var_mask = batch['var_mask'].to(device)
        mpm_mask = batch['mpm_mask'].to(device)

        loss, mpm, var, dom = model(waveform, waveform_var, domain_id, var_mask, mpm_mask)

        total_loss += loss.item()
        total_mpm += mpm.item()
        total_var += var.item()
        total_dom += dom.item()

    model.train()
    
    num_batches = len(dataloader)
    return (
        total_loss / num_batches,
        total_mpm / num_batches,
        total_var / num_batches,
        total_dom / num_batches
    )

# ------------------------------
# 7. prepare_dataloader_from_npz - 讀取 npz 檔案並準備 DataLoader
# ------------------------------
def prepare_dataloader_from_npz(npz_path, batch_size = 32, train_ratio = 0.7, val_ratio = 0.1, seed = 42):
    """
    npz_path: 儲存 waveforms 與 metadata 的 npz 檔案路徑
    batch_size: 每個批次的大小
    train_ratio: 訓練集比例
    val_ratio: 驗證集比例
    測試集比例會是 1 - (train_ratio + val_ratio)
    """
    data = np.load(npz_path)
    waveforms = data['waveforms']
    metadata = data['metadata']
    # 先分離出測試集 (比例 = 1 - (train_ratio + val_ratio))
    test_size = 1 - (train_ratio + val_ratio)
    X_train_val, X_test, meta_train_val, meta_test = train_test_split(
        waveforms, metadata, test_size = test_size, random_state = seed, shuffle = True
    )
    # 再從剩下的資料中分離驗證集，比例調整為 val_ratio / (train_ratio + val_ratio)
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, meta_train, meta_val = train_test_split(
        X_train_val, meta_train_val, test_size = val_ratio_adjusted, random_state = seed, shuffle = True
    )

    train_set = SeismicDataset(X_train, meta_train)
    val_set = SeismicDataset(X_val, meta_val)
    test_set = SeismicDataset(X_test, meta_test)

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader, test_loader
# ------------------------------
# 8. Entry Point - 主要訓練流程的入口
# ------------------------------
if __name__ == '__main__':
    # 選擇設備：若有 GPU 則使用 cuda，否則使用 cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 從 npz 檔案中準備 DataLoader
    train_loader, val_loader, test_loader = prepare_dataloader_from_npz("stead_combined_chunks.npz", batch_size=32, train_ratio=0.7, val_ratio=0.1)
    # 初始化模型
    model = TimesBERTForSeismic()
    # 執行訓練（每個 epoch 用驗證集評估）
    train_model(model, train_loader, val_loader, device, num_epochs=100, lr=1e-4)
    # 訓練完成後，用測試集進行最終評估
    test_loss, test_mpm, test_var, test_dom = evaluate_model(model, test_loader, device)
    print(f"🧪 Final Test Performance - Loss: {test_loss:.4f}, MPM: {test_mpm:.4f}, VAR: {test_var:.4f}, DOM: {test_dom:.4f}")
