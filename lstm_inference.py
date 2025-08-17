import os
import glob
import json
import math
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import from existing modules
from forcasting_lstm.model import ForecastingModel

# TJU 데이터의 컬럼 순서 (매우 중요 - 절대 변경하지 말 것)
TJU_COLUMNS = [
    'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
    'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
    'current mean', 'current std', 'current kurtosis', 'current skewness', 
    'CV Q', 'CV charge time', 'current slope', 'current entropy',
    'capacity'
]

def load_model_and_stats(model_path: str, device: str = "cpu") -> Tuple[nn.Module, Dict]:
    """학습된 모델과 정규화 통계량을 로드"""
    
    # 모델 파일에서 컬럼명 추출
    base_name = os.path.basename(model_path)
    if base_name.startswith('lstm_'):
        column_name = base_name[5:]  # 'lstm_' 제거
        if column_name.endswith('.pth'):
            column_name = column_name[:-4]  # '.pth' 제거
    else:
        raise ValueError(f"모델 파일명 형식이 올바르지 않습니다: {base_name}")
    
    # 체크포인트에서 모델 구조 정보 추출
    checkpoint = torch.load(model_path, map_location=device)
    
    # 체크포인트에서 모델 크기 추론
    weight_ih_l0_shape = checkpoint.get('backbone.weight_ih_l0', None)
    weight_hh_l0_shape = checkpoint.get('backbone.weight_hh_l0', None)
    head_weight_shape = checkpoint.get('head.weight', None)
    
    if weight_ih_l0_shape is not None and weight_hh_l0_shape is not None and head_weight_shape is not None:
        # LSTM의 hidden_size 추론 (weight_hh의 두 번째 차원)
        hidden_size = weight_hh_l0_shape.shape[1]
        
        # layers 수 추론 (weight_ih_l{i} 키의 개수로 추정)
        layer_keys = [k for k in checkpoint.keys() if k.startswith('backbone.weight_ih_l')]
        layers = len(layer_keys)
        
        print(f"  추론된 모델 구조 - hidden_size: {hidden_size}, layers: {layers}")
        
        # 모델 초기화 (파이프라인에서 사용한 하이퍼파라미터와 일치시킴)
        model = ForecastingModel(
            input_size=1,
            hidden_size=hidden_size,  # 추론된 값 사용
            layers=layers,            # 추론된 값 사용
            dropout=0.1,              # 파이프라인 기본값
            backbone='lstm'
        )
    else:
        print(f"  경고: 모델 구조를 추론할 수 없습니다. 파이프라인 기본값 사용")
        # 파이프라인 기본 하이퍼파라미터 사용
        model = ForecastingModel(
            input_size=1,
            hidden_size=64,   # 파이프라인 기본값
            layers=2,         # 파이프라인 기본값
            dropout=0.1,      # 파이프라인 기본값
            backbone='lstm'
        )
    
    # 모델 가중치 로드
    try:
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print(f"  모델 로드 성공: {column_name}")
    except Exception as e:
        raise RuntimeError(f"모델 로드 실패 ({column_name}): {e}")
    
    # 정규화 통계량은 별도 파일에서 로드해야 할 수 있음
    # 여기서는 기본값 사용 (실제로는 학습 시 저장된 값을 사용해야 함)
    stats = {
        'mu': 0.0,  # 실제 학습 시 저장된 평균값 사용
        'std': 1.0,  # 실제 학습 시 저장된 표준편차 사용
        'column': column_name
    }
    
    return model, stats

def predict_sequence(
    model: nn.Module, 
    initial_sequence: np.ndarray,
    n_steps: int,
    mu: float,
    std: float,
    window: int = 1000,
    device: str = "cpu"
) -> np.ndarray:
    """순차적으로 시계열 예측 수행"""
    
    # 정규화
    sequence_norm = (initial_sequence - mu) / std
    
    # 초기 컨텍스트 설정 (마지막 window 길이만큼)
    if len(sequence_norm) >= window:
        context = sequence_norm[-window:].copy()
    else:
        # 시퀀스가 window보다 짧은 경우 전체 사용
        context = sequence_norm.copy()
        # window 길이에 맞추기 위해 패딩 (앞쪽을 첫 번째 값으로 채움)
        if len(context) < window:
            padding = np.full(window - len(context), context[0])
            context = np.concatenate([padding, context])
    
    predictions_norm = []
    
    model.eval()
    with torch.no_grad():
        for _ in range(n_steps):
            # 마지막 window 길이만큼을 입력으로 사용
            x = torch.from_numpy(context[-window:]).reshape(1, window, 1).to(device)
            pred = model(x).cpu().numpy().ravel()[0]
            predictions_norm.append(pred)
            
            # 컨텍스트 업데이트
            context = np.concatenate([context, np.array([pred], dtype=np.float32)])
    
    # 역정규화
    predictions = np.array(predictions_norm) * std + mu
    return predictions

def load_tju_file(file_path: str) -> pd.DataFrame:
    """TJU 데이터 파일을 로드하고 컬럼 순서 확인"""
    
    df = pd.read_csv(file_path)
    
    # 컬럼 순서 확인 및 조정
    if list(df.columns) != TJU_COLUMNS:
        print(f"경고: {file_path}의 컬럼 순서가 예상과 다릅니다.")
        print(f"파일 컬럼: {list(df.columns)}")
        print(f"예상 컬럼: {TJU_COLUMNS}")
        
        # 컬럼이 존재하는지 확인하고 순서 맞추기
        missing_cols = set(TJU_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")
        
        # 컬럼 순서 맞추기
        df = df[TJU_COLUMNS]
    
    return df

def estimate_normalization_stats(data: np.ndarray, window: int = 1000) -> Tuple[float, float]:
    """데이터의 초기 부분을 사용해 정규화 통계량 추정"""
    if len(data) >= window:
        base_data = data[:window]
    else:
        base_data = data
    
    mu = np.mean(base_data)
    std = np.std(base_data)
    if std == 0:
        std = 1.0
    
    return float(mu), float(std)

def get_relative_path(file_path: str, base_dir: str) -> str:
    """기준 디렉토리로부터의 상대 경로를 반환"""
    # 절대 경로로 변환
    file_abs = os.path.abspath(file_path)
    base_abs = os.path.abspath(base_dir)
    
    # 상대 경로 계산
    rel_path = os.path.relpath(file_abs, base_abs)
    
    # 파일명 제거하고 디렉토리만 반환
    rel_dir = os.path.dirname(rel_path)
    
    return rel_dir

def main():
    parser = argparse.ArgumentParser(description='TJU 데이터 예측')
    parser.add_argument('--tju_data_dir', type=str, default='data/TJU data/', 
                       help='TJU 데이터 디렉토리')
    parser.add_argument('--model_dir', type=str, default='forcasting_lstm/lstm_models/',
                       help='학습된 모델들이 있는 디렉토리')
    parser.add_argument('--output_dir', type=str, default='predictions/TJU_predictions/',
                       help='예측 결과를 저장할 디렉토리')
    parser.add_argument('--window', type=int, default=15,
                       help='입력 윈도우 크기')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='학습에 사용할 데이터 비율 (나머지는 예측)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='사용할 디바이스')
    parser.add_argument('--target_columns', type=str, nargs='+', 
                       default=['voltage_mean', 'voltage_std', 'voltage_kurtosis', 'voltage_skewness',
                                'CC_Q', 'CC_charge_time', 'voltage_slope', 'voltage_entropy',
                                'current_mean', 'current_std', 'current_kurtosis', 'current_skewness', 
                                'CV_Q', 'CV_charge_time', 'current_slope', 'current_entropy'],
                       help='예측할 대상 컬럼들 (capacity는 제외됨 - 참조용으로만 사용)')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # TJU 데이터 파일들 찾기
    tju_pattern = os.path.join(args.tju_data_dir, "**", "*.csv")
    tju_files = glob.glob(tju_pattern, recursive=True)
    
    if not tju_files:
        raise ValueError(f"TJU 데이터 파일을 찾을 수 없습니다: {tju_pattern}")
    
    print(f"발견된 TJU 파일 수: {len(tju_files)}")
    
    # 사용 가능한 모델들 로드
    available_models = {}
    for col in args.target_columns:
        model_path = os.path.join(args.model_dir, f"lstm_{col}.pth")
        if os.path.exists(model_path):
            try:
                model, stats = load_model_and_stats(model_path, args.device)
                available_models[col] = {'model': model, 'stats': stats}
                print(f"모델 로드 성공: {col}")
            except Exception as e:
                print(f"모델 로드 실패 ({col}): {e}")
        else:
            print(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    if not available_models:
        raise ValueError("사용 가능한 모델이 없습니다.")
    
    # 각 TJU 파일에 대해 예측 수행
    results_summary = []
    
    for tju_file in sorted(tju_files):
        print(f"\n처리 중: {tju_file}")
        
        try:
            # 데이터 로드
            df = load_tju_file(tju_file)
            
            # 파일의 상대 경로 구조 가져오기
            rel_path = get_relative_path(tju_file, args.tju_data_dir)
            
            # 출력 디렉토리에 동일한 구조 생성
            output_subdir = os.path.join(args.output_dir, rel_path)
            os.makedirs(output_subdir, exist_ok=True)
            
            # 파일명 정보
            file_name = os.path.basename(tju_file)
            dataset_name = os.path.basename(os.path.dirname(tju_file))
            
            print(f"  - 데이터셋: {dataset_name}")
            print(f"  - 저장 경로: {output_subdir}")
            
            # 학습/예측 구간 분할
            total_length = len(df)
            train_end_idx = int(total_length * args.train_ratio)
            
            if train_end_idx < args.window:
                print(f"  경고: 파일이 너무 짧습니다 (길이: {total_length}, 필요: {args.window})")
                continue
            
            predict_steps = total_length - train_end_idx
            if predict_steps <= 0:
                print(f"  경고: 예측할 구간이 없습니다.")
                continue
            
            print(f"  - 총 길이: {total_length}")
            print(f"  - 학습 구간: 0 ~ {train_end_idx-1}")
            print(f"  - 예측 구간: {train_end_idx} ~ {total_length-1}")
            print(f"  - 예측 스텝: {predict_steps}")
            
            # 전체 데이터를 포함한 결과 데이터프레임 생성
            predictions_df = pd.DataFrame()
            predictions_df['index'] = np.arange(total_length)
            predictions_df['data_type'] = ['train'] * train_end_idx + ['predict'] * predict_steps
            
            for col_name, model_info in available_models.items():
                # 컬럼명을 TJU 데이터 형식으로 변환
                if col_name == 'CC_Q':
                    tju_col = 'CC Q'
                elif col_name == 'CC_charge_time':
                    tju_col = 'CC charge time'
                elif col_name == 'CV_Q':
                    tju_col = 'CV Q'
                elif col_name == 'CV_charge_time':
                    tju_col = 'CV charge time'
                elif col_name == 'voltage_mean':
                    tju_col = 'voltage mean'
                elif col_name == 'voltage_std':
                    tju_col = 'voltage std'
                elif col_name == 'voltage_kurtosis':
                    tju_col = 'voltage kurtosis'
                elif col_name == 'voltage_skewness':
                    tju_col = 'voltage skewness'
                elif col_name == 'voltage_slope':
                    tju_col = 'voltage slope'
                elif col_name == 'voltage_entropy':
                    tju_col = 'voltage entropy'
                elif col_name == 'current_mean':
                    tju_col = 'current mean'
                elif col_name == 'current_std':
                    tju_col = 'current std'
                elif col_name == 'current_kurtosis':
                    tju_col = 'current kurtosis'
                elif col_name == 'current_skewness':
                    tju_col = 'current skewness'
                elif col_name == 'current_slope':
                    tju_col = 'current slope'
                elif col_name == 'current_entropy':
                    tju_col = 'current entropy'
                else:
                    tju_col = col_name
                
                if tju_col not in df.columns:
                    print(f"    경고: 컬럼 '{tju_col}'을 찾을 수 없습니다.")
                    continue
                
                # 해당 컬럼의 전체 실제값
                true_values_full = df[tju_col].values.astype(np.float32)
                
                # 해당 컬럼의 학습 데이터
                train_data = true_values_full[:train_end_idx]
                
                # 정규화 통계량 추정 (실제로는 학습 시 저장된 값 사용해야 함)
                mu, std = estimate_normalization_stats(train_data, args.window)
                
                # 예측 수행
                preds = predict_sequence(
                    model=model_info['model'],
                    initial_sequence=train_data,
                    n_steps=predict_steps,
                    mu=mu,
                    std=std,
                    window=args.window,
                    device=args.device
                )
                
                # 전체 결과 조합: 학습구간은 실제값, 예측구간은 예측값
                combined_values = np.concatenate([train_data, preds])
                
                # 원본 컬럼명으로 저장 (예측값)
                predictions_df[tju_col] = combined_values
                
                # 예측 구간만의 실제값
                true_values_predict = true_values_full[train_end_idx:]
                
                # 메트릭 계산 (예측 구간만)
                rmse = math.sqrt(np.mean((true_values_predict - preds) ** 2))
                mae = np.mean(np.abs(true_values_predict - preds))
                mape = np.mean(np.abs((true_values_predict - preds) / (true_values_predict + 1e-8))) * 100
                
                print(f"    {col_name}: RMSE={rmse:.6f}, MAE={mae:.6f}, MAPE={mape:.2f}%")
                
                # 결과 요약에 추가
                results_summary.append({
                    'file': file_name,
                    'dataset': dataset_name,
                    'relative_path': rel_path,
                    'column': col_name,
                    'train_length': train_end_idx,
                    'predict_length': predict_steps,
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape
                })
            
            # capacity 원본값 추가 (예측하지 않고 참조용으로만)
            if 'capacity' in df.columns:
                predictions_df['capacity'] = df['capacity'].values
            
            # 예측 결과 저장 (폴더 구조 유지)
            base_name = os.path.splitext(file_name)[0]
            output_file = os.path.join(output_subdir, f"pred_{base_name}.csv")
            predictions_df.to_csv(output_file, index=False)
            print(f"  저장완료: {output_file}")
            
        except Exception as e:
            print(f"  오류 발생: {e}")
            continue
    
    # 전체 결과 요약 저장
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_file = os.path.join(args.output_dir, "prediction_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        
        # 컬럼별 평균 성능
        print("\n=== 전체 결과 요약 ===")
        for col in args.target_columns:
            col_results = summary_df[summary_df['column'] == col]
            if len(col_results) > 0:
                avg_rmse = col_results['rmse'].mean()
                avg_mae = col_results['mae'].mean()
                avg_mape = col_results['mape'].mean()
                print(f"{col}: 평균 RMSE={avg_rmse:.6f}, MAE={avg_mae:.6f}, MAPE={avg_mape:.2f}%")
        
        print(f"\n전체 요약 저장: {summary_file}")
    
    print(f"\n모든 예측 결과가 '{os.path.abspath(args.output_dir)}'에 저장되었습니다.")
    print("폴더 구조:")
    for root, dirs, files in os.walk(args.output_dir):
        level = root.replace(args.output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.csv'):
                print(f"{subindent}{file}")

if __name__ == "__main__":
    main()