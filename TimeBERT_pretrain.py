import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class SeismicDataset(Dataset):
    def __init__(self, waveform_list, metadata, patch_size = 100, mask_ratio = 0.25):
        """
        waveform_list: 波形資料列表，形狀為 (N, 3, T) 代表 N 筆資料，每筆有 3 個變數、T 個時間點
        metadata: 每筆資料的其他元資料，形狀為 (N, 7)，其中第 4、5 欄為經緯度
        patch_size: 分割波形片段的大小
        mask_ratio: 在 MPM (Masked Patch Modeling) 中遮蔽片段的比例
        """
        self.le = LabelEncoder()
        self.waveforms = waveform_list  # 原始波形資料
        self.metadata = metadata        # 對應的元資料（包含經緯度資訊）
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.domain_ids = self.le.fit_transform(self.metadata['station_code'])  # 根據Station Code，組合成站點
        self.pga = metadata['trace_pga_cmps2'].values.astype('float32')

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
        # # --- 加入標準化處理 --- #
        # 針對每個 channel 計算均值與標準差（沿時間軸 T）
        mean = waveform.mean(dim=1, keepdim=True)
        std = waveform.std(dim=1, keepdim=True) + 1e-5  # 加入 epsilon 防止除以 0
        waveform = (waveform - mean) / std
        # ------------------------ #
        domain_id = torch.tensor(self.domain_ids[idx], dtype=torch.long)        # 對應的 domain_id
        C, T = waveform.shape
        pga = torch.tensor(self.pga[idx], dtype=torch.float32)

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

        # # 假設替換完後，只重新標準化被替換的 channel
        # channel_data = waveform_for_var[replace_idx]
        # channel_mean = channel_data.mean()
        # channel_std = channel_data.std() + 1e-5
        # waveform_for_var[replace_idx] = (channel_data - channel_mean) / channel_std

        num_patches = T // self.patch_size
        mpm_mask = torch.rand(C, num_patches) < self.mask_ratio

        return {
            'waveform': waveform,          # 原始輸入 → MPM / DOM 用
            'waveform_var': waveform_for_var,     # 替換後 → VAR 任務專用
            'domain_id': domain_id,
            'var_mask': var_mask,
            'mpm_mask': mpm_mask,
            'pga': pga
        }

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=100, embed_dim=768, num_domains=1000):
        """
        patch_size: 每個 patch 的時間點數量
        embed_dim: 嵌入向量的維度
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # 線性層將每個 patch 投影到嵌入空間
        self.linear = nn.Linear(patch_size, embed_dim)

        # 建立每個變量（總共 3 個）的嵌入向量
        self.var_embed = nn.Embedding(3, embed_dim)

        # 為每個站點（domain）建立嵌入，這裡設定最大站點數 1000
        self.dom_embed = nn.Embedding(num_domains, embed_dim)

        # 位置編碼（learnable）
        self.pos_embed = nn.Parameter(torch.randn(1, 384, embed_dim))  # shape: (1, num_patches, embed_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, waveform, domain_ids):
        """
        waveform: 輸入波形資料，形狀 (B, C, T)
        domain_ids: 每筆序列對應的站點 id，形狀 (B,)
        """
        B, C, T = waveform.shape
        num_patches = T // self.patch_size

        # 擷取完整的 patch 部分並重塑為 (B, C, num_patches, patch_size)
        patches = waveform[:, :, :num_patches * self.patch_size].reshape(B, C, num_patches, self.patch_size)
        
        # 線性投影得到 patch tokens，形狀 (B, C, num_patches, embed_dim)
        tokens = self.linear(patches)

        # 加上位置編碼（broadcast 到 C channel）：(B, C, num_patches, embed_dim)
        tokens = tokens + self.pos_embed[:, :num_patches, :].unsqueeze(1)  # unsqueeze(1) → (1, 1, num_patches, D)

        # 為每個變量加入專屬的 token（形狀從 (C, embed_dim) 擴充到 (B, C, embed_dim)）
        var_tokens = self.var_embed(torch.arange(C, device=waveform.device))
        var_tokens = var_tokens.unsqueeze(0).expand(B, -1, -1)

        # 為每個序列加入站點 token
        dom_tokens = self.dom_embed(domain_ids)
        return tokens, var_tokens, dom_tokens, patches

class TimesBERTEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_layers=6):
        """
        embed_dim: 輸入 token 的嵌入維度
        num_layers: Transformer 層數
        """
        super().__init__()
        # 定義單層 Transformer 編碼器層（注意頭數與內部維度）
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=8, dim_feedforward=2048)
        # 堆疊多層 Transformer 編碼器
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # 輸入序列需轉換成 Transformer 所要求的 shape
        return self.encoder(x)

class FTPHead(nn.Module):
    def __init__(self, embed_dim, num_domains):
        """
        embed_dim: 輸入 token 的嵌入維度
        num_domains: 站點總數
        """
        super().__init__()
        # 用於變量分類（例如原始與替換）
        self.var_classifier = nn.Linear(embed_dim, 2)
        # 用於站點分類
        self.dom_classifier = nn.Linear(embed_dim, num_domains)

    def forward(self, z_var, z_dom, var_labels, dom_labels):
        # 計算變量分類損失
        loss_var = nn.functional.cross_entropy(
            self.var_classifier(z_var.reshape(-1, z_var.size(-1))), 
            var_labels.reshape(-1).long()
        )
        # 計算站點分類損失
        loss_dom = nn.functional.cross_entropy(self.dom_classifier(z_dom), dom_labels.long())
        return loss_var + loss_dom, loss_var, loss_dom

class TimesBERTForSeismic(nn.Module):
    def __init__(self, patch_size=100, embed_dim=768, num_domains=1000, loss_weights=(1.0, 2.0, 5.0)):
        """
        patch_size: 每個 patch 的大小
        embed_dim: token 嵌入的維度
        num_domains: 站點（domain）的數量
        """
        super().__init__()
        self.patch_size = patch_size
        self.embedding = PatchEmbedding(patch_size, embed_dim, num_domains)
        self.encoder = TimesBERTEncoder(embed_dim)
        self.ftp = FTPHead(embed_dim, num_domains)
        # 為 MPM (Masked Patch Modeling) 建立投影頭，將編碼器輸出投影回原始 patch 大小
        self.projection_head = nn.Linear(embed_dim, patch_size)
        self.loss_weights = loss_weights  # (lambda_mpm, lambda_var, lambda_dom)

    def forward(self, waveform, waveform_var, domain_ids, var_mask, mpm_mask):
        """
        waveform: 輸入波形，形狀 (B, C, T)
        waveform_var: 變量替換後的波形（VAR任務專用）
        domain_ids: 每筆資料對應的站點 id，形狀 (B,)
        var_mask: 變量替換的遮罩，形狀 (B, C)
        mpm_mask: MPM 遮罩，形狀 (B, C, num_patches)
        """
        B, C, T = waveform.shape
        # 執行 patch embedding，取得 tokens 與特殊 token 以及原始 patch 資料
        tokens, var_tokens, dom_tokens, original_patches = self.embedding(waveform, domain_ids)
        
        # 對 tokens 做遮蔽（MPM）：將被遮蔽的 tokens 設為 0
        tokens_masked = tokens.clone()
        mask_expanded = mpm_mask.unsqueeze(-1).expand_as(tokens_masked)
        
        # mask_token = 0
        # tokens_masked[mask_expanded] = 0

        # mask_token = 可學習參數
        tokens_masked[mask_expanded] = self.embedding.mask_token.expand_as(tokens_masked[mask_expanded])

        # 組合所有 tokens：先放入 DOM token，再加入所有 patch token，最後加入 VAR token
        all_tokens = torch.cat([
            dom_tokens.unsqueeze(1),                           # (B, 1, embed_dim)
            tokens_masked.view(B, -1, tokens.size(-1)),          # (B, C*num_patches, embed_dim)
            var_tokens                                          # (B, C, embed_dim)
        ], dim=1)

        # 編碼器處理，轉換成 Transformer 所需的 shape
        encoded = self.encoder(all_tokens.transpose(0, 1)).transpose(0, 1)
        # 取得 DOM token 的編碼結果
        z_dom = encoded[:, 0, :]

        # 對替換後的 waveform（VAR 任務專用）進行 embedding 並獲取新的 VAR token
        _, var_tokens_alt, _, _ = self.embedding(waveform_var, domain_ids)
        all_tokens_var = torch.cat([
            dom_tokens.unsqueeze(1),
            tokens.view(B, -1, tokens.size(-1)),
            var_tokens_alt
        ], dim=1)
        encoded_var = self.encoder(all_tokens_var.transpose(0, 1)).transpose(0, 1)
        # 取得 VAR token 的編碼結果
        z_var = encoded_var[:, -C:, :]

        # 重塑 patch token 的部分
        patch_token_start = 1
        patch_token_end = 1 + C * tokens.size(2)
        encoded_patches = encoded[:, patch_token_start:patch_token_end, :].view(B, C, -1, encoded.size(-1))
        # 投影回原始 patch 尺寸
        pred_patches = self.projection_head(encoded_patches)

        # 用 reshape 解決 indexing，將最後一維保留
        masked_pred = pred_patches[mpm_mask]
        masked_true = original_patches[mpm_mask]

        if masked_pred.numel() > 0:
            masked_pred = masked_pred.view(-1, self.patch_size)
            masked_true = masked_true.view(-1, self.patch_size)
            mpm_loss = F.mse_loss(masked_pred, masked_true)
        else:
            mpm_loss = torch.tensor(0.0, device=waveform.device)

        ftp_loss, loss_var, loss_dom = self.ftp(z_var, z_dom, var_mask, domain_ids)
        lambda_mpm, lambda_var, lambda_dom = self.loss_weights
        total_loss = lambda_mpm * mpm_loss + lambda_var * loss_var + lambda_dom * loss_dom

        return total_loss, lambda_mpm * mpm_loss, lambda_var * loss_var, lambda_dom * loss_dom
