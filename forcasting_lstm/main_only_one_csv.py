from utils import SeqDataset, UniSeqDataset
from model import ForecastingModel

# main.py
import argparse
import os
import math
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error

# -----------------------------
# 유틸: 표준화
# -----------------------------
def fit_standardizer(df):
    mu = df.mean(axis=0)
    std = df.std(axis=0)
    if isinstance(std, (pd.Series, pd.DataFrame)):
        std = std.replace(0, 1.0)
    else:
        # numpy scalar일 때
        if std == 0:
            std = 1.0
    return mu, std

def apply_standardizer(df: pd.DataFrame, mu: pd.Series, std: pd.Series):
    return (df - mu) / std

# -----------------------------
# 학습/평가 루프
# -----------------------------
def train_one(
    model: nn.Module,
    dl_tr: DataLoader,
    dl_va: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
):
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_hist, va_hist = [], []

    model.to(device)
    for ep in range(1, epochs + 1):
        # train
        model.train()
        run = 0.0
        n = 0
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(-1) if yb.ndim == 1 else yb  # [B,1] or [B,T]
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            run += loss.item() * xb.size(0)
            n += xb.size(0)
        tr_loss = run / max(1, n)
        tr_hist.append(tr_loss)

        # val
        model.eval()
        run = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device)
                yb = yb.to(device).unsqueeze(-1) if yb.ndim == 1 else yb
                pred = model(xb)
                loss = crit(pred, yb)
                run += loss.item() * xb.size(0)
                n += xb.size(0)
        va_loss = run / max(1, n)
        va_hist.append(va_loss)

        print(f"[{ep:03d}/{epochs}] train={tr_loss:.6f}  val={va_loss:.6f}")

    return tr_hist, va_hist

def evaluate(model: nn.Module, dl_te: DataLoader, device: str = "cpu"):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            pred = model(xb.to(device)).cpu().numpy()
            y = yb.numpy()
            # pred shape: [B,1] or [B,T]
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred[:, 0]
            preds.append(pred)
            trues.append(y)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    if preds.ndim == 1:
        rmse = math.sqrt(mean_squared_error(trues, preds))
    else:
        rmse = math.sqrt(mean_squared_error(trues, preds))
    return rmse, trues, preds

# -----------------------------
# 메인
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, default="../data/hust_data/1-1.csv")
    ap.add_argument("--target_col", type=str, default="CC Q")
    ap.add_argument("--window", type=int, default = 1000)       # <= "첫 100행" 요구에 맞춤
    ap.add_argument("--n_future", type=int, default=-1)      # -1이면 파일 끝까지 예측
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", type=str, default='lstm')
    ap.add_argument("--out_dir", type=str, default="./runs_uni")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 1) 데이터 로드
    df = pd.read_csv(args.csv_path)
    if args.target_col not in df.columns:
        raise ValueError(f"'{args.target_col}' 컬럼을 못 찾았습니다. CSV 컬럼들: {list(df.columns)}")

    # 시간 정렬(있으면)
    if "time/s" in df.columns:
        df = df.sort_values("time/s").reset_index(drop=True)

    series = df[args.target_col].to_numpy().astype(np.float32)
    N = len(series)
    if N <= args.window:
        raise ValueError(f"행이 {N}개인데 window={args.window}라서 학습할 윈도우가 없습니다. window를 줄이세요.")

    # 2) 정규화는 "초기 window 구간"의 통계로만 수행 (실제 운용 가정)
    base = series[:args.window]       # 첫 100행
    mu, std = fit_standardizer(base)
    series_norm = (series - mu) / std

    # 3) 학습 데이터: 0..(window) 구간에서 생성 가능한 모든 (입력윈도우, 다음스텝) 쌍
    #    마지막 타깃 인덱스 = window (즉 101번째 행)까지의 타깃이 존재
    train_end_idx = args.window       # 포함 인덱스
    ds = UniSeqDataset(series_norm, window=args.window, train_end_idx=train_end_idx)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False)

    # 4) 모델/학습
    device = torch.device(args.device)
    model = ForecastingModel(
        input_size = 1,
        hidden_size = args.hidden_size,
        layers = args.layers,
        dropout = args.dropout,
        backbone = args.backbone
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()

    model.train()
    for ep in range(1, args.epochs + 1):
        run = 0.0
        n = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            run += loss.item() * xb.size(0)
            n += xb.size(0)
        print(f"[{ep:03d}/{args.epochs}] train_mse={run/max(1,n):.6f}")

    # 5) 순차(autoregressive) 예측 시작
    #    시작 컨텍스트: 마지막 window 구간(인덱스 [window-window .. window-1]) = 0..window-1
    context = series_norm[:args.window].copy()  # 길이 window
    preds_norm = []

    # 예측할 스텝 수 결정
    if args.n_future is None or args.n_future < 0:
        # 파일 끝까지의 실제 길이를 기준으로: (N - window) 스텝
        steps = N - args.window
    else:
        steps = int(args.n_future)

    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            x = torch.from_numpy(context[-args.window:]).reshape(1, args.window, 1).to(device)
            yhat = model(x).cpu().numpy().ravel()[0]  # 정규화된 예측
            preds_norm.append(yhat)
            # 다음 예측을 위해 예측값을 컨텍스트 뒤에 붙임
            context = np.concatenate([context, np.array([yhat], dtype=np.float32)], axis=0)

    preds = np.array(preds_norm) * std + mu

    # 6) 평가(가능하면)
    metrics = {}
    if steps > 0 and (args.window + steps) <= N:
        # 실제 구간: series[window : window+steps]
        y_true = series[args.window : args.window + steps]
        rmse = math.sqrt(mean_squared_error(y_true, preds))
        metrics["rmse"] = float(rmse)
        print(f"Test RMSE (orig units) on next {steps} steps: {rmse:.6f}")
    else:
        y_true = None
        print(f"(주의) 실제값과의 비교 구간이 없습니다. steps={steps}, N={N}")

    # 7) 결과 저장
    out = pd.DataFrame({
        "index": np.arange(args.window, args.window + steps),
        "pred": preds,
    })
    if y_true is not None:
        out["true"] = y_true

    out_csv = os.path.join(args.out_dir, f"forecast_{args.target_col.replace('/','_')}.csv")
    out.to_csv(out_csv, index=False)
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{args.target_col.replace('/','_')}.pt"))

    print(f"\nSaved forecast to: {os.path.abspath(out_csv)}")
    if metrics:
        print("Metrics:", metrics)


if __name__ == "__main__":
    main()