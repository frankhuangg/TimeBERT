# TimeBERT.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=100, embed_dim=768):
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
        self.dom_embed = nn.Embedding(1000, embed_dim)

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
    def __init__(self, patch_size=100, embed_dim=768, num_domains=1000):
        """
        patch_size: 每個 patch 的大小
        embed_dim: token 嵌入的維度
        num_domains: 站點（domain）的數量
        """
        super().__init__()
        self.patch_size = patch_size
        self.embedding = PatchEmbedding(patch_size, embed_dim)
        self.encoder = TimesBERTEncoder(embed_dim)
        self.ftp = FTPHead(embed_dim, num_domains)
        # 為 MPM (Masked Patch Modeling) 建立投影頭，將編碼器輸出投影回原始 patch 大小
        self.projection_head = nn.Linear(embed_dim, patch_size)

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
        tokens_masked[mask_expanded] = 0

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
        if mpm_mask.sum() > 0:
            mpm_loss = nn.functional.mse_loss(pred_patches[mpm_mask], original_patches[mpm_mask])
        else:
            mpm_loss = 0.0

        ftp_loss, loss_var, loss_dom = self.ftp(z_var, z_dom, var_mask, domain_ids)

        return mpm_loss + ftp_loss, mpm_loss, loss_var, loss_dom
