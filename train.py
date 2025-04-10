# train.py
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from TimeBERT_pretrain import TimesBERTForSeismic
from dataloader import prepare_dataloader_from_npz

def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=1e-4):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    # 用於紀錄各個 epoch 的損失
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

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            waveform = batch['waveform'].to(device)
            domain_id = batch['domain_id'].to(device)
            var_mask = batch['var_mask'].to(device)
            mpm_mask = batch['mpm_mask'].to(device)
            waveform_var = batch['waveform_var'].to(device)

            optimizer.zero_grad()
            loss, mpm_loss, var_loss, dom_loss = model(waveform, waveform_var, domain_id, var_mask, mpm_mask)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mpm_loss += mpm_loss.item()
            total_var_loss += var_loss.item()
            total_dom_loss += dom_loss.item()

        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_mpm_loss = total_mpm_loss / num_batches
        avg_var_loss = total_var_loss / num_batches
        avg_dom_loss = total_dom_loss / num_batches

        epoch_total_loss.append(avg_loss)
        epoch_mpm_loss.append(avg_mpm_loss)
        epoch_var_loss.append(avg_var_loss)
        epoch_dom_loss.append(avg_dom_loss)
        print(f"✅ Epoch {epoch+1} - Avg Total Loss: {avg_loss:.4f} | MPM: {avg_mpm_loss:.4f}, VAR: {avg_var_loss:.4f}, DOM: {avg_dom_loss:.4f}")

        # Evaluate on validation set
        val_loss, val_mpm, val_var, val_dom = evaluate_model(model, val_loader, device)
        print(f"🔍 Val   - Avg Loss: {val_loss:.4f} | MPM: {val_mpm:.4f}, VAR: {val_var:.4f}, DOM: {val_dom:.4f}")

        val_total_loss.append(val_loss)
        val_mpm_loss.append(val_mpm)
        val_var_loss.append(val_var)
        val_dom_loss.append(val_dom)

    # 繪製訓練與驗證損失曲線
    epochs = range(1, num_epochs + 1)
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

if __name__ == '__main__':
    # 選擇運行設備（若有 GPU 則使用 cuda）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 從 npz 檔案中準備資料 DataLoader
    train_loader, val_loader, test_loader = prepare_dataloader_from_npz(
        "stead_combined_chunks.npz", batch_size=32, train_ratio=0.7, val_ratio=0.1
    )
    # 載入模型
    model = TimesBERTForSeismic()
    # 執行訓練
    train_model(model, train_loader, val_loader, device, num_epochs=100, lr=1e-4)
    # 訓練完成後用測試集做最終評估
    test_loss, test_mpm, test_var, test_dom = evaluate_model(model, test_loader, device)
    print(f"🧪 Final Test Performance - Loss: {test_loss:.4f}, MPM: {test_mpm:.4f}, VAR: {test_var:.4f}, DOM: {test_dom:.4f}")
