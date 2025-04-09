# train_cwa.py

import time
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchaudio # 確保已安裝: pip install torchaudio

# 從您的 dataloader 和 model 文件導入
from CWA_dataloader import cwa_loader, SELECTED_CWA_ATTRS, collate_fn_skip_none
from model_seismic_clip_two_branch import AUDIO_CLIP # 假設您有這個文件
from utils_cwa import accuracy # 假設您有 utils_cwa.py

import argparse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='SeisCLIP CWA Pretraining - GPU Preprocessing & Regularization')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size (RTX 4090: 128, 256, ...)')
    parser.add_argument('--num_workers', type=int, default=0, help='Dataloader workers (Windows: 0)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Maximum number of epochs')
    # --- 調整超參數 ---
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Adjusted learning rate (e.g., 5e-5)')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for regularization (e.g., 1e-4)')
    # --- 早停 ---
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Epochs to wait for improvement before stopping')
    # --- 路徑 ---
    parser.add_argument('--save_interval', type=int, default=10, help='interval for model saving (epochs)')
    parser.add_argument('--plot_save_path', type=str, default='./plots_cwa_gpu_stft/', help='Path to save training plots')
    parser.add_argument('--model_save_path', type=str, default='./work_dir_cwa/model_SeisClip_cwa_20s_gpu_stft/', help='path to save the model')
    parser.add_argument('--metadata_train_csv', type=str, default="D:/shihan/pretrain_data_all_channels_accel/train.csv", help='Path to processed train metadata CSV')
    parser.add_argument('--hdf5_train_path', type=str, default="D:/shihan/pretrain_data_all_channels_accel/train.hdf5", help='Path to processed train HDF5 file')
    parser.add_argument('--metadata_valid_csv', type=str, default="D:/shihan/pretrain_data_all_channels_accel/valid.csv", help='Path to processed valid metadata CSV')
    parser.add_argument('--hdf5_valid_path', type=str, default="D:/shihan/pretrain_data_all_channels_accel/valid.hdf5", help='Path to processed valid HDF5 file')
    # --- 波形/STFT ---
    parser.add_argument('--segment_length', type=int, default=2000, help='Waveform segment length (samples)')
    parser.add_argument('--p_arrival_offset', type=int, default=500, help='P-arrival offset in segment (samples)')
    parser.add_argument('--sample_rate', type=int, default=100, help='Sample rate (Hz)')
    parser.add_argument('--stft_n_fft', type=int, default=100, help='STFT n_fft')
    parser.add_argument('--stft_win_length', type=int, default=20, help='STFT win_length')
    parser.add_argument('--stft_hop_length', type=int, default=10, help='STFT hop_length')
    # --- 模型 ---
    parser.add_argument('--embed_dim', type=int, default=384, help='Embedding dimension for CLIP')
    parser.add_argument('--text_input_dim', type=int, default=len(SELECTED_CWA_ATTRS), help='Dimension of text/info input vector')
    parser.add_argument('--text_width', type=int, default=512, help='Hidden width in InfoEncoder MLP')
    parser.add_argument('--text_layers', type=int, default=2, help='Number of hidden layers in InfoEncoder MLP')
    parser.add_argument('--spec_fdim', type=int, default=50, help='Spectrum frequency dimension (n_fft // 2)')
    parser.add_argument('--spec_model_size', type=str, default='small224', help='AST model size variant')
    parser.add_argument('--imagenet_pretrain', type=bool, default=True, help='Use ImageNet pretraining for AST')

    args = parser.parse_args()
    return args

# --- GPU STFT 輔助函數 (與上次相同) ---
def calculate_gpu_stft(waveform_batch, n_fft, hop_length, win_length, device):
    """ 在 GPU 上計算 STFT 並返回所需的頻譜圖 """
    B, C, T_wave = waveform_batch.shape
    window = torch.hann_window(win_length, device=device)
    waveform_reshaped = waveform_batch.view(B * C, T_wave)
    stft_result = torch.stft(
        waveform_reshaped, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window=window, center=True,
        pad_mode='constant', normalized=False, onesided=True,
        return_complex=True
    )
    spectrogram_abs = torch.abs(stft_result)
    spectrogram_abs = spectrogram_abs[:, 1:, :] # Remove DC
    _, Freq, T_spec = spectrogram_abs.shape
    spectrogram_final = spectrogram_abs.view(B, C, Freq, T_spec)
    return spectrogram_final

def train(args):
    # ... (初始化 seed, path, device) ...
    set_seed(args.seed)
    if not os.path.exists(args.model_save_path): os.makedirs(args.model_save_path)
    if not os.path.exists(args.plot_save_path): os.makedirs(args.plot_save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 創建 Dataloader (返回波形段) ---
    print("創建 Dataloader...")
    # ... (創建 train_dataset, val_dataset - 與上次相同) ...
    try:
        train_dataset = cwa_loader(
            metadata_csv_path=args.metadata_train_csv,
            hdf5_path=args.hdf5_train_path,
            segment_length=args.segment_length,
            p_arrival_offset=args.p_arrival_offset,
        )
        val_dataset = cwa_loader(
            metadata_csv_path=args.metadata_valid_csv,
            hdf5_path=args.hdf5_valid_path,
            segment_length=args.segment_length,
            p_arrival_offset=args.p_arrival_offset,
        )
    except Exception as e:
        print(f"初始化 Dataloader 時發生錯誤: {e}"); return
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("錯誤：訓練或驗證數據集為空。"); return


    # --- 動態計算 spec_tdim ---
    print("計算 GPU STFT 的時間維度...")
    spec_time_dim = 0
    # ... (計算 spec_time_dim - 與上次相同) ...
    if len(train_dataset) > 0:
        temp_info, temp_waveform = None, None
        for i in range(min(len(train_dataset), 100)): # Check first 100 samples
            sample = train_dataset[i]
            if sample is not None:
                temp_info, temp_waveform = sample; break
        if temp_waveform is not None:
            with torch.no_grad(): # No need to track gradients here
                 temp_waveform_batch = temp_waveform.unsqueeze(0).to(device)
                 temp_spec = calculate_gpu_stft(temp_waveform_batch, args.stft_n_fft, args.stft_hop_length, args.stft_win_length, device)
                 spec_time_dim = temp_spec.shape[-1]
            print(f"計算得到的 GPU STFT 時間維度 (spec_tdim): {spec_time_dim}")
            del temp_waveform_batch, temp_spec; torch.cuda.empty_cache() # Clean up memory
        else: print("錯誤：無法從數據集獲取樣本來計算 STFT 維度。"); return
    else: print("錯誤：訓練數據集為空。"); return
    if spec_time_dim <= 0: print("錯誤：未能計算出有效的 spec_time_dim。"); return


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn_skip_none, pin_memory=torch.cuda.is_available())
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_skip_none, pin_memory=torch.cuda.is_available())

    # --- 創建模型 ---
    print("創建模型...")
    # ... (創建模型 - 與上次相同，使用計算出的 spec_time_dim) ...
    model = AUDIO_CLIP(
        embed_dim=args.embed_dim, text_input=args.text_input_dim,
        text_width=args.text_width, text_layers=args.text_layers,
        spec_fdim=args.spec_fdim, spec_tdim=spec_time_dim,
        spec_model_size=args.spec_model_size, device_name=str(device),
        imagenet_pretrain=args.imagenet_pretrain
    ).to(device)

    # --- 優化器 (使用調整後的 LR 和 Weight Decay) ---
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True) # patience=5

    # --- (早停變數, history - 與上次相同) ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc1': [], 'val_acc1': [], 'train_acc5': [], 'val_acc5': []}

    print("開始訓練 (GPU Preprocessing, LR={}, WD={})...".format(args.learning_rate, args.weight_decay))
    start_time_total = time.time()
    # --- 主訓練循環 ---
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0; train_acc1 = 0.0; train_acc5 = 0.0
        processed_batches_train = 0

        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} Train", leave=False)
        for batch_idx, batch_data in enumerate(train_pbar):
            if batch_data is None or batch_data[0] is None: continue
            info_batch, waveform_batch = batch_data
            info = info_batch.to(device, non_blocking=True)
            waveform = waveform_batch.to(device, non_blocking=True)
            labels = torch.arange(info.size(0), device=device)

            # === GPU 預處理 ===
            try:
                 mean = waveform.mean(dim=-1, keepdim=True); std = waveform.std(dim=-1, keepdim=True)
                 waveform_norm = (waveform - mean) / (std + 1e-6)
                 spec = calculate_gpu_stft(waveform_norm, args.stft_n_fft, args.stft_hop_length, args.stft_win_length, device)
                 if model.training: # 只在訓練時應用增強
                    try:
                          # === 修正後的初始化 ===
                          # freq_masking_param: 頻率遮罩的最大寬度 (多少個連續的頻率 bin)
                          # time_masking_param: 時間遮罩的最大寬度 (多少個連續的時間步)
                          # n_freq_masks:       應用多少個頻率遮罩
                          # n_time_masks:       應用多少個時間遮罩
                          spec_augment = torchaudio.transforms.SpecAugment(
                              freq_mask_param=5,  # 範例值: 最多遮蓋 5 個頻率 bin
                              time_mask_param=20, # 範例值: 最多遮蓋 20 個時間步 (約 0.2 秒)
                              n_freq_masks=1,        # 應用 1 個頻率遮罩
                              n_time_masks=1         # 應用 1 個時間遮罩
                          ).to(device) # 將轉換本身也放到 GPU 上執行效率更高
                          spec = spec_augment(spec) # 應用數據增強
                    except Exception as aug_e:
                        print(f"\nError during SpecAugment: {aug_e}")
                        # 如果增強出錯，可以選擇跳過這個 batch 或使用未增強的 spec
                        continue # 這裡選擇跳過
                 if spec.shape[-1] != spec_time_dim or spec.shape[-2] != args.spec_fdim: continue # Skip if shape mismatch
            except Exception as e: print(f"\nGPU Preprocessing Error (train): {e}"); continue

            # === 模型計算 ===
            try:
                (features), logits, loss = model(info, spec)
            except Exception as e: print(f"\nModel Forward/Loss Error (train): {e}"); continue

            # --- (反向傳播, 優化, 記錄 loss/acc - 與上次相同) ---
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            # ... (記錄 loss, acc, 更新 pbar) ...
            train_loss += loss.item(); acc1, acc5 = accuracy(logits, labels, topk=(1, 5)); train_acc1 += acc1.item(); train_acc5 += acc5.item(); processed_batches_train += 1
            train_pbar.set_postfix(loss=f"{train_loss / processed_batches_train:.4f}", acc1=f"{train_acc1 / processed_batches_train:.2f}%")

        train_pbar.close()
        if processed_batches_train == 0: print(f"\nEpoch {epoch+1}: No batches processed in training."); continue
        # ... (計算 epoch 平均值, 存入 history - 與上次相同) ...
        avg_train_loss = train_loss / processed_batches_train; avg_train_acc1 = train_acc1 / processed_batches_train; avg_train_acc5 = train_acc5 / processed_batches_train
        history['train_loss'].append(avg_train_loss); history['train_acc1'].append(avg_train_acc1); history['train_acc5'].append(avg_train_acc5)

        # --- 驗證 ---
        model.eval()
        val_loss = 0.0; val_acc1 = 0.0; val_acc5 = 0.0
        processed_batches_val = 0
        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} Valid", leave=False)
        with torch.no_grad():
            for batch_data in val_pbar:
                 if batch_data is None or batch_data[0] is None: continue
                 info_batch, waveform_batch = batch_data
                 info = info_batch.to(device, non_blocking=True)
                 waveform = waveform_batch.to(device, non_blocking=True)
                 labels = torch.arange(info.size(0), device=device)

                 # === GPU 預處理 ===
                 try:
                     mean = waveform.mean(dim=-1, keepdim=True); std = waveform.std(dim=-1, keepdim=True)
                     waveform_norm = (waveform - mean) / (std + 1e-6)
                     spec = calculate_gpu_stft(waveform_norm, args.stft_n_fft, args.stft_hop_length, args.stft_win_length, device)
                     if spec.shape[-1] != spec_time_dim or spec.shape[-2] != args.spec_fdim: continue
                 except Exception as e: print(f"\nGPU Preprocessing Error (validation): {e}"); continue

                 # === 模型計算 ===
                 try:
                     (features), logits, loss = model(info, spec)
                 except Exception as e: print(f"\nModel Forward/Loss Error (validation): {e}"); continue

                 # --- (記錄 loss/acc - 與上次相同) ---
                 val_loss += loss.item(); acc1, acc5 = accuracy(logits, labels, topk=(1, 5)); val_acc1 += acc1.item(); val_acc5 += acc5.item(); processed_batches_val += 1
                 val_pbar.set_postfix(loss=f"{val_loss / processed_batches_val:.4f}", acc1=f"{val_acc1 / processed_batches_val:.2f}%")

        val_pbar.close()
        
        # --- 計算 epoch 平均值並存儲 ---
        if processed_batches_val == 0:
            print(f"\nEpoch {epoch+1}: No batches processed in validation.") # 加換行符
            avg_val_loss = float('inf') # 設為 inf 以便早停邏輯正常工作
            avg_val_acc1 = 0.0
            avg_val_acc5 = 0.0
        else:
            avg_val_loss = val_loss / processed_batches_val
            avg_val_acc1 = val_acc1 / processed_batches_val
            avg_val_acc5 = val_acc5 / processed_batches_val

        # 存儲 history
        history['val_loss'].append(avg_val_loss)
        history['val_acc1'].append(avg_val_acc1)
        history['val_acc5'].append(avg_val_acc5)

        # *** 在打印之前計算 epoch_time ***
        epoch_time = time.time() - epoch_start_time

        # --- 打印 epoch 摘要 ---
        # 檢查 avg_val_loss 是否是 inf 或 NaN (如果驗證完全失敗)
        val_loss_str = f"{avg_val_loss:.4f}" if avg_val_loss != float('inf') and not np.isnan(avg_val_loss) else "N/A"
        print(f'\n====> Epoch: {epoch+1}/{args.num_epochs} | Time: {epoch_time:.2f}s') # 使用 \n 確保換行
        print(f'    Train Loss: {avg_train_loss:.4f} | Acc@1: {avg_train_acc1:.2f}% | Acc@5: {avg_train_acc5:.2f}%')
        print(f'    Valid Loss: {val_loss_str} | Acc@1: {avg_val_acc1:.2f}% | Acc@5: {avg_val_acc5:.2f}%')

        # --- 更新學習率, 檢查早停, 保存模型 (這部分邏輯不變) ---
        # 注意: scheduler.step() 需要一個數值，不能是 inf
        if avg_val_loss != float('inf') and not np.isnan(avg_val_loss):
            scheduler.step(avg_val_loss)

            # 檢查早停 (只有在 val_loss 有效時才檢查)
            if avg_val_loss < best_val_loss:
                print(f"    Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # --- (保存 best_model_state 的代碼) ---
                best_model_state = {
                    'epoch': epoch + 1, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'best_val_loss': best_val_loss, 'args': args, 'spec_time_dim': spec_time_dim
                }
                save_path_best = os.path.join(args.model_save_path, 'model_best.pth.tar')
                torch.save(best_model_state, save_path_best)
                # ---
            else:
                epochs_no_improve += 1
                print(f"    Validation loss did not improve for {epochs_no_improve} / {args.early_stopping_patience} epochs.")

            if epochs_no_improve >= args.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break # 跳出訓練循環
        else:
             # 如果驗證損失無效，也增加 No Improve 計數，避免無限循環
             epochs_no_improve += 1
             print(f"    Validation loss is invalid. Not improving for {epochs_no_improve} / {args.early_stopping_patience} epochs.")
             if epochs_no_improve >= args.early_stopping_patience:
                  print(f"\nEarly stopping triggered due to invalid validation loss after {epoch + 1} epochs.")
                  break

        # --- 定期保存檢查點 (可選) ---
        # if (epoch + 1) % args.save_interval == 0:
        #     save_checkpoint = { ... } # 可以保存當前狀態，不一定是最佳狀態
        #     save_path = os.path.join(args.model_save_path, f'checkpoint_epoch_{epoch+1}.pth.tar')
        #     torch.save(save_checkpoint, save_path)

    # --- 訓練結束 ---
    total_training_time = time.time() - start_time_total
    print(f"訓練完成！總耗時: {total_training_time / 3600:.2f} 小時")
    print(f"最佳驗證損失: {best_val_loss:.4f}")

    # --- 繪圖 ---
    # 確保 history 列表長度一致
    num_actual_epochs = len(history['train_loss'])
    epochs_range = range(1, num_actual_epochs + 1)

    plt.figure(figsize=(12, 10))

    # Loss 圖
    plt.subplot(2, 1, 1)
    if num_actual_epochs > 0:
        plt.plot(epochs_range, history['train_loss'], label='Training Loss')
        plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy 圖
    plt.subplot(2, 1, 2)
    if num_actual_epochs > 0:
        plt.plot(epochs_range, history['train_acc1'], label='Training Accuracy @1', linestyle='-')
        plt.plot(epochs_range, history['val_acc1'], label='Validation Accuracy @1', linestyle='-')
        plt.plot(epochs_range, history['train_acc5'], label='Training Accuracy @5', linestyle=':')
        plt.plot(epochs_range, history['val_acc5'], label='Validation Accuracy @5', linestyle=':')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy (Top-1 & Top-5)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_filename = os.path.join(args.plot_save_path, 'training_curves.png')
    try:
        plt.savefig(plot_filename)
        print(f"訓練曲線圖已保存至: {plot_filename}")
    except Exception as e:
        print(f"保存繪圖時發生錯誤: {e}")
    # plt.show()

# --- 省略 main 函數 (與上次相同) ---
def main():
    args = parse_args()
    print("Arguments:", args)
    if args.text_input_dim != len(SELECTED_CWA_ATTRS):
         print(f"Warning: --text_input_dim ({args.text_input_dim}) != len(SELECTED_CWA_ATTRS) ({len(SELECTED_CWA_ATTRS)}). Check SELECTED_CWA_ATTRS in dataloader.")
         args.text_input_dim = len(SELECTED_CWA_ATTRS)
         print(f"Corrected --text_input_dim to {args.text_input_dim}")

    train(args)

if __name__ == '__main__':
    main()