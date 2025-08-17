import torch
import numpy as np

from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(self, df_part, feat_cols, target_cols, window=60, horizon=1, max_samples=4000):
        self.X = df_part[feat_cols].values.astype(np.float32)
        self.Y = df_part[target_cols].values.astype(np.float32)
        self.window = window
        self.horizon = horizon
        self.indices = []
        last_start = len(self.X) - (window + horizon) + 1
        for s in range(max(0, last_start)):
            self.indices.append(s)
        # Subsample to speed up demo
        if len(self.indices) > max_samples:
            step = len(self.indices) // max_samples
            self.indices = self.indices[::max(1, step)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        e = s + self.window
        h = e + self.horizon - 1
        x = self.X[s:e]
        y = self.Y[h]
        return torch.from_numpy(x), torch.from_numpy(y)

class UniSeqDataset(Dataset):
    """
    1D 시계열(단일 컬럼)에서 window 길이로 입력, 다음 1스텝을 타깃으로 학습.
    데이터는 0..(train_end_idx) 범위에서만 윈도우를 구성합니다.
    """
    def __init__(self, series: np.ndarray, window: int, train_end_idx: int):
        """
        series: (N,) 1D numpy array (정규화된 값)
        window: 입력 길이
        train_end_idx: 학습에 사용할 마지막 인덱스(포함). 보통 window-1 이상이어야 함.
                       예: window=100이면, 0..99가 마지막 입력 스텝. 타깃은 1..100.
        """
        self.x = series.astype(np.float32)
        self.window = window
        self.end = train_end_idx
        assert self.end >= window - 1, "train_end_idx는 최소 window-1 이상이어야 합니다."
        # 가능한 시작 인덱스 s: 입력 x[s:s+window]의 마지막 시점은 s+window-1 ≤ end-1 (타깃이 end까지 있어야 함)
        # 타깃 y는 x[(s+window)] 위치
        self.indices = []
        # 마지막 s는 (end - window) 까지 가능(타깃이 end 위치까지 도달)
        for s in range(0, self.end - self.window + 1):
            self.indices.append(s)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        e = s + self.window
        x = self.x[s:e]                # [window]
        y = self.x[e]                  # 다음 1스텝
        return torch.from_numpy(x).unsqueeze(-1), torch.tensor([y], dtype=torch.float32)