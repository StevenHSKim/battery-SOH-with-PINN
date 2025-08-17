from utils import SeqDataset, UniSeqDataset
from model import ForecastingModel

# main_multi.py
import argparse
import os
import math
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import mean_squared_error
import glob
from typing import List, Dict, Tuple

# -----------------------------
# 유틸: 표준화
# -----------------------------
def fit_standardizer_multi(data_list: List[np.ndarray]) -> Tuple[float, float]:
    """여러 시계열 데이터에서 전체 통계량 계산"""
    all_data = np.concatenate(data_list, axis=0)
    mu = np.mean(all_data)
    std = np.std(all_data)
    if std == 0:
        std = 1.0
    return mu, std

def apply_standardizer(data: np.ndarray, mu: float, std: float) -> np.ndarray:
    return (data - mu) / std

# -----------------------------
# 다중 파일 데이터셋 클래스
# -----------------------------
class MultiFileUniSeqDataset:
    """여러 CSV 파일의 시계열 데이터를 하나의 데이터셋으로 통합"""
    
    def __init__(self, csv_paths: List[str], target_col: str, window: int, train_end_ratio: float = 0.7):
        self.csv_paths = csv_paths
        self.target_col = target_col
        self.window = window
        self.train_end_ratio = train_end_ratio
        
        # 각 파일별 데이터 로드 및 전처리
        self.file_data = {}
        self.train_data_list = []
        
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            if target_col not in df.columns:
                raise ValueError(f"'{target_col}' 컬럼을 {csv_path}에서 찾을 수 없습니다.")
            
            # 시간 정렬(있으면)
            if "time/s" in df.columns:
                df = df.sort_values("time/s").reset_index(drop=True)
            
            series = df[target_col].to_numpy().astype(np.float32)
            N = len(series)
            
            if N <= window:
                print(f"경고: {csv_path}의 행이 {N}개로 window({window})보다 작습니다. 건너뜀.")
                continue
            
            # 학습 구간 결정
            train_end_idx = int(N * train_end_ratio)
            if train_end_idx <= window:
                train_end_idx = window + 1
            
            # 학습용 데이터 추출 (정규화는 나중에)
            train_portion = series[:train_end_idx]
            self.train_data_list.append(train_portion)
            
            # 파일별 전체 데이터 저장 (추론용)
            self.file_data[csv_path] = {
                'series': series,
                'train_end_idx': train_end_idx,
                'total_length': N
            }
        
        # 전체 학습 데이터를 기반으로 정규화 파라미터 계산
        self.mu, self.std = fit_standardizer_multi(self.train_data_list)
        
        print(f"로드된 파일 수: {len(self.file_data)}")
        print(f"전체 정규화 파라미터 - 평균: {self.mu:.6f}, 표준편차: {self.std:.6f}")
    
    def get_train_dataset(self) -> UniSeqDataset:
        """학습용 통합 데이터셋 반환"""
        # 모든 학습 데이터를 정규화하고 통합
        all_normalized = []
        for train_data in self.train_data_list:
            normalized = apply_standardizer(train_data, self.mu, self.std)
            all_normalized.append(normalized)
        
        # 통합 데이터 생성
        combined_data = np.concatenate(all_normalized, axis=0)
        
        # UniSeqDataset으로 변환 (전체 데이터를 학습에 사용)
        dataset = UniSeqDataset(combined_data, window=self.window, train_end_idx=len(combined_data)-1)
        
        return dataset

# -----------------------------
# 학습 함수
# -----------------------------
def train_model(
    model: nn.Module,
    dl_tr: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """모델 학습"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    
    model.to(device)
    model.train()
    
    for ep in range(1, epochs + 1):
        run = 0.0
        n = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            run += loss.item() * xb.size(0)
            n += xb.size(0)
        
        avg_loss = run / max(1, n)
        print(f"[{ep:03d}/{epochs}] train_mse={avg_loss:.6f}")

# -----------------------------
# 추론 함수
# -----------------------------
def forecast_single_file(
    model: nn.Module,
    series: np.ndarray,
    window: int,
    train_end_idx: int,
    mu: float,
    std: float,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """단일 파일에 대한 순차 예측"""
    
    # 정규화
    series_norm = apply_standardizer(series, mu, std)
    
    # 학습 구간의 마지막 window 길이만큼을 초기 컨텍스트로 사용
    context = series_norm[train_end_idx-window:train_end_idx].copy()
    
    # 예측할 스텝 수
    total_length = len(series)
    steps = total_length - train_end_idx
    
    if steps <= 0:
        return np.array([]), np.array([])
    
    preds_norm = []
    model.eval()
    
    with torch.no_grad():
        for _ in range(steps):
            # 마지막 window 길이만큼 입력으로 사용
            x = torch.from_numpy(context[-window:]).reshape(1, window, 1).to(device)
            yhat = model(x).cpu().numpy().ravel()[0]
            preds_norm.append(yhat)
            
            # 컨텍스트 업데이트
            context = np.concatenate([context, np.array([yhat], dtype=np.float32)], axis=0)
    
    # 역정규화
    preds = np.array(preds_norm) * std + mu
    y_true = series[train_end_idx:train_end_idx + steps]
    
    return preds, y_true

# -----------------------------
# 메인 함수
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", type=str, default =  "../data/hust_data/", help="CSV 파일들이 있는 디렉토리")
    ap.add_argument("--csv_pattern", type=str, default="*.csv", help="CSV 파일 패턴")
    ap.add_argument("--target_col", type=str, default="CC Q")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--train_ratio", type=float, default=0.7, help="학습에 사용할 비율")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", type=str, default='lstm')
    ap.add_argument("--out_dir", type=str, default="./runs_multi")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # 시드 설정
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # CSV 파일 목록 가져오기
    csv_pattern = os.path.join(args.csv_dir, args.csv_pattern)
    csv_paths = glob.glob(csv_pattern)
    
    if not csv_paths:
        raise ValueError(f"'{csv_pattern}' 패턴과 일치하는 CSV 파일을 찾을 수 없습니다.")
    
    print(f"발견된 CSV 파일 수: {len(csv_paths)}")
    for path in sorted(csv_paths):
        print(f"  - {path}")

    # 1) 다중 파일 데이터셋 생성
    multi_dataset = MultiFileUniSeqDataset(
        csv_paths=csv_paths,
        target_col=args.target_col,
        window=args.window,
        train_end_ratio=args.train_ratio
    )
    
    # 2) 학습용 데이터로더 생성
    train_dataset = multi_dataset.get_train_dataset()
    dl_train = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=False)
    
    print(f"학습 데이터셋 크기: {len(train_dataset)}")

    # 3) 모델 생성 및 학습
    device = torch.device(args.device)
    model = ForecastingModel(
        input_size=1,
        hidden_size=args.hidden_size,
        layers=args.layers,
        dropout=args.dropout,
        backbone=args.backbone
    ).to(device)

    print("\n=== 모델 학습 시작 ===")
    train_model(
        model=model,
        dl_tr=dl_train,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device
    )

    # 4) 각 파일별 추론 및 평가
    print("\n=== 파일별 추론 및 평가 ===")
    all_results = {}
    
    for csv_path in multi_dataset.file_data.keys():
        file_info = multi_dataset.file_data[csv_path]
        file_name = os.path.basename(csv_path)
        
        print(f"\n처리 중: {file_name}")
        
        # 예측 수행
        preds, y_true = forecast_single_file(
            model=model,
            series=file_info['series'],
            window=args.window,
            train_end_idx=file_info['train_end_idx'],
            mu=multi_dataset.mu,
            std=multi_dataset.std,
            device=args.device
        )
        
        if len(preds) == 0:
            print(f"  - 예측할 구간이 없습니다.")
            continue
        
        # 평가
        rmse = math.sqrt(mean_squared_error(y_true, preds))
        mae = np.mean(np.abs(y_true - preds))
        mape = np.mean(np.abs((y_true - preds) / y_true)) * 100
        
        print(f"  - 학습 구간: 0 ~ {file_info['train_end_idx']-1}")
        print(f"  - 예측 구간: {file_info['train_end_idx']} ~ {file_info['total_length']-1}")
        print(f"  - 예측 스텝 수: {len(preds)}")
        print(f"  - RMSE: {rmse:.6f}")
        print(f"  - MAE: {mae:.6f}")
        print(f"  - MAPE: {mape:.2f}%")
        
        # 결과 저장
        result_df = pd.DataFrame({
            'index': np.arange(file_info['train_end_idx'], file_info['train_end_idx'] + len(preds)),
            'true': y_true,
            'pred': preds,
            'error': y_true - preds,
            'abs_error': np.abs(y_true - preds)
        })
        
        # 파일명에서 확장자 제거하고 저장
        base_name = os.path.splitext(file_name)[0]
        out_csv = os.path.join(args.out_dir, f"forecast_{base_name}.csv")
        result_df.to_csv(out_csv, index=False)
        
        all_results[file_name] = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'n_predictions': len(preds),
            'train_end_idx': file_info['train_end_idx']
        }

    # 5) 전체 결과 요약 및 저장
    print("\n=== 전체 결과 요약 ===")
    if all_results:
        avg_rmse = np.mean([r['rmse'] for r in all_results.values()])
        avg_mae = np.mean([r['mae'] for r in all_results.values()])
        avg_mape = np.mean([r['mape'] for r in all_results.values()])
        
        print(f"전체 평균 RMSE: {avg_rmse:.6f}")
        print(f"전체 평균 MAE: {avg_mae:.6f}")
        print(f"전체 평균 MAPE: {avg_mape:.2f}%")
        
        # 결과 요약을 JSON으로 저장
        summary = {
            'config': vars(args),
            'normalization': {'mu': float(multi_dataset.mu), 'std': float(multi_dataset.std)},
            'avg_metrics': {'rmse': avg_rmse, 'mae': avg_mae, 'mape': avg_mape},
            'file_results': all_results
        }
        
        with open(os.path.join(args.out_dir, 'results_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    # 모델 저장
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model_multi.pt'))
    
    print(f"\n모든 결과가 '{os.path.abspath(args.out_dir)}'에 저장되었습니다.")


if __name__ == "__main__":
    main()