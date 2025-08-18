#!/usr/bin/env python3
"""
LSTM 결과를 사용한 PINN 추론 코드
1. LSTM 예측 결과 CSV 파일들을 로드 (Dataset별 폴더 구조 지원)
2. Dataset별로 다른 PINN 모델 사용
3. 데이터 형식: index,data_type,16개특징,capacity
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import glob
import json
import math
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# PINN 관련 import
from Model.Model import PINN
from utils.util import eval_metrix

# TJU 데이터의 특징 컬럼 순서 (capacity 제외한 16개) - 실제 데이터 순서와 일치
FEATURE_COLUMNS = [
    'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
    'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
    'current mean', 'current std', 'current kurtosis', 'current skewness', 
    'CV Q', 'CV charge time', 'current slope', 'current entropy'
]

# Dataset별 PINN 모델 매핑
DATASET_MODEL_MAPPING = {
    'Dataset_1_NCA_battery': 'model_TJU_0.pth',
    'Dataset_2_NCM_battery': 'model_TJU_1.pth', 
    'Dataset_3_NCM_NCA_battery': 'model_TJU_2.pth'
}

# Dataset별 nominal capacity 매핑 (TJUdata 클래스와 동일)
DATASET_NOMINAL_CAPACITY = {
    'Dataset_1_NCA_battery': 3.5,
    'Dataset_2_NCM_battery': 3.5,
    'Dataset_3_NCM_NCA_battery': 2.5
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def apply_pinn_normalization(features: np.ndarray, normalization_method: str = 'min-max') -> np.ndarray:
    """PINN 학습 시와 동일한 정규화 적용"""
    
    if normalization_method == 'min-max':
        # Min-Max 정규화: [-1, 1] 범위로 변환
        normalized = 2 * (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0)) - 1
    elif normalization_method == 'z-score':
        # Z-score 정규화
        normalized = (features - features.mean(axis=0)) / features.std(axis=0)
    else:
        normalized = features
    
    return normalized.astype(np.float32)

def load_original_tju_dataset(data_dir: str, train_ratio: float = 0.7) -> Dict[str, List[Dict]]:
    """원본 TJU 데이터셋 로드 및 학습/예측 구간 분할"""
    
    dataset_data = {}
    
    # Dataset 폴더들 찾기
    for dataset_name in DATASET_MODEL_MAPPING.keys():
        dataset_path = os.path.join(data_dir, dataset_name)
        
        if not os.path.exists(dataset_path):
            print(f"Dataset 폴더를 찾을 수 없습니다: {dataset_path}")
            continue
            
        print(f"\n=== {dataset_name} 로딩 중 ===")
        
        # 해당 Dataset 폴더의 모든 CSV 파일들 찾기
        csv_pattern = os.path.join(dataset_path, "*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            print(f"  {dataset_name}에서 CSV 파일을 찾을 수 없습니다.")
            continue
        
        print(f"  발견된 파일: {len(csv_files)}개")
        
        dataset_files = []
        
        for csv_file in sorted(csv_files):
            print(f"  로딩 중: {os.path.basename(csv_file)}")
            
            try:
                df = pd.read_csv(csv_file)
                
                # 필요한 컬럼들이 있는지 확인
                required_columns = FEATURE_COLUMNS + ['capacity']
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    print(f"    경고: 누락된 컬럼들: {missing_cols}")
                    continue
                
                # 전체 데이터 길이
                total_length = len(df)
                if total_length < 10:  # 최소 길이 체크
                    print(f"    경고: 데이터가 너무 짧습니다 ({total_length}개)")
                    continue
                
                # 학습/예측 구간 분할
                train_end_idx = int(total_length * train_ratio)
                
                # index와 data_type 컬럼 추가
                df['index'] = np.arange(total_length)
                df['data_type'] = ['train'] * train_end_idx + ['predict'] * (total_length - train_end_idx)
                
                # 예측 구간 추출
                predict_rows = df[df['data_type'] == 'predict'].copy()
                
                if len(predict_rows) == 0:
                    print(f"    경고: 예측 구간이 없습니다.")
                    continue
                
                # 특징 데이터 추출 (16개 컬럼)
                features = predict_rows[FEATURE_COLUMNS].values.astype(np.float32)  # [N, 16]
                capacity_true = predict_rows['capacity'].values.astype(np.float32)  # [N,]
                
                # PINN 학습 시와 동일한 정규화 적용
                features_normalized = apply_pinn_normalization(features, 'min-max')
                
                # cycle 인덱스 생성 (예측 구간의 실제 인덱스 사용)
                cycle_indices = predict_rows['index'].values.astype(np.float32).reshape(-1, 1)  # [N, 1]
                
                # PINN 입력 형태로 결합: [정규화된 features + cycle] -> [N, 17]
                pinn_input = np.hstack([features_normalized, cycle_indices])
                
                # 전체 DataFrame을 임시 파일로 저장 (save_predictions에서 사용)
                temp_file_path = csv_file + "_processed.csv"
                df.to_csv(temp_file_path, index=False)
                
                dataset_files.append({
                    'file': temp_file_path,  # 처리된 DataFrame 경로
                    'original_file': csv_file,  # 원본 파일 경로
                    'filename': os.path.basename(csv_file),
                    'dataset': dataset_name,
                    'pinn_input': pinn_input,
                    'capacity_true': capacity_true,
                    'num_cycles': len(features),
                    'original_indices': predict_rows['index'].values,
                    'train_end_idx': train_end_idx,
                    'total_length': total_length
                })
                
                print(f"    로드 완료: {len(features_normalized)}개 예측 사이클 (정규화 적용)")
                print(f"    학습 구간: 0~{train_end_idx-1}, 예측 구간: {train_end_idx}~{total_length-1}")
                
            except Exception as e:
                print(f"    오류 발생: {e}")
                continue
        
        if dataset_files:
            dataset_data[dataset_name] = dataset_files
            print(f"  {dataset_name}: 총 {len(dataset_files)}개 파일 로드 완료")
        else:
            print(f"  {dataset_name}: 로드된 파일이 없습니다.")
    
    return dataset_data

def create_pinn_args(args):
    """PINN 모델용 arguments 생성"""
    pinn_args = argparse.Namespace()
    
    # 기본 설정 (원본 PINN 코드와 동일하게 유지)
    pinn_args.data = 'TJU'
    pinn_args.batch_size = args.batch_size
    pinn_args.normalization_method = 'min-max'
    
    # 추론 전용 설정 (학습 관련 파라미터는 최소값)
    pinn_args.epochs = 1
    pinn_args.early_stop = 1
    pinn_args.warmup_epochs = 1
    pinn_args.warmup_lr = 0.002
    pinn_args.lr = 0.01
    pinn_args.final_lr = 0.0002
    pinn_args.lr_F = 0.001
    
    # 모델 구조 (원본과 동일)
    pinn_args.F_layers_num = 3
    pinn_args.F_hidden_dim = 60
    
    # 손실 함수 가중치 (원본과 동일)
    pinn_args.alpha = 1.0
    pinn_args.beta = 0.05
    
    # 로그 및 저장 경로
    pinn_args.log_dir = 'pinn_inference.txt'
    pinn_args.save_folder = args.output_dir
    
    return pinn_args

def load_pinn_model_for_dataset(dataset_name: str, model_dir: str, args) -> PINN:
    """Dataset에 맞는 PINN 모델 로드"""
    
    if dataset_name not in DATASET_MODEL_MAPPING:
        raise ValueError(f"알 수 없는 Dataset: {dataset_name}")
    
    model_filename = DATASET_MODEL_MAPPING[dataset_name]
    model_path = os.path.join(model_dir, model_filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PINN 모델 파일을 찾을 수 없습니다: {model_path}")
    
    # PINN arguments 생성
    pinn_args = create_pinn_args(args)
    
    # PINN 모델 초기화
    pinn_model = PINN(pinn_args)
    
    # 사전 훈련된 가중치 로드
    print(f"  {dataset_name}용 PINN 모델 로드 중: {model_filename}")
    pinn_model.load_model(model_path)
    
    # 추론 모드로 설정
    pinn_model.eval()
    
    print(f"  {dataset_name} PINN 모델 로드 완료!")
    return pinn_model

def predict_capacity_with_pinn(pinn_model: PINN, pinn_input: np.ndarray) -> np.ndarray:
    """PINN 모델로 예측 (capacity 또는 SOH)"""
    
    # numpy to torch tensor
    input_tensor = torch.FloatTensor(pinn_input).to(device)
    
    # 예측 수행
    pinn_model.eval()
    with torch.no_grad():
        pred = pinn_model.predict(input_tensor)
        pred = pred.cpu().numpy().flatten()
    
    return pred

def evaluate_predictions(capacity_true: np.ndarray, capacity_pred: np.ndarray) -> Dict:
    """예측 성능 평가"""
    
    # utils.util의 eval_metrix 함수 사용
    mae, mape, mse, rmse = eval_metrix(capacity_pred, capacity_true)
    
    # MAPE가 이미 퍼센트(%)인지 확인하고 조정
    # eval_metrix가 0~1 범위로 반환하면 100을 곱해야 함
    if mape < 1.0:  # 0~1 범위로 추정되면 퍼센트로 변환
        mape_percent = float(mape * 100)
        print(f"    MAPE를 퍼센트로 변환: {mape:.6f} → {mape_percent:.2f}%")
    else:  # 이미 퍼센트로 추정
        mape_percent = float(mape)
        print(f"    MAPE가 이미 퍼센트: {mape_percent:.2f}%")
    
    return {
        'MAE': float(mae),
        'MAPE': mape_percent,  # 항상 퍼센트로 반환
        'MSE': float(mse),
        'RMSE': float(rmse)
    }

def save_predictions(data_item: Dict, soh_pred: np.ndarray, metrics: Dict, output_dir: str):
    """예측 결과 저장 - 원본 CSV에 true_soh, soh_pred 컬럼 추가"""
    
    # Dataset별 폴더 생성
    dataset_output_dir = os.path.join(output_dir, data_item['dataset'])
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # 처리된 CSV 파일 로드 (index, data_type 컬럼 포함)
    processed_df = pd.read_csv(data_item['file'])
    
    # 초기 capacity (첫 번째 행의 capacity 값)
    initial_capacity = processed_df['capacity'].iloc[0]
    
    # true_soh 계산: capacity / 초기 capacity
    processed_df['true_soh'] = processed_df['capacity'] / initial_capacity
    
    # soh_pred 컬럼 추가 (전체에 NaN으로 초기화)
    processed_df['soh_pred'] = np.nan
    
    # 예측 구간(data_type == 'predict')에만 soh_pred 값 할당
    predict_mask = processed_df['data_type'] == 'predict'
    processed_df.loc[predict_mask, 'soh_pred'] = soh_pred  # PINN이 예측한 SOH
    
    # 파일명 생성 (원본 파일명에 pinn_ 접두사 추가)
    base_name = data_item['filename']
    output_filename = f"pinn_{base_name}"
    output_path = os.path.join(dataset_output_dir, output_filename)
    
    # 전체 CSV 저장 (원본 + index + data_type + true_soh + soh_pred 컬럼)
    processed_df.to_csv(output_path, index=False)
    
    # 임시 파일 정리
    if os.path.exists(data_item['file']) and data_item['file'].endswith('_processed.csv'):
        os.remove(data_item['file'])
    
    # 메트릭 정보를 JSON으로 저장
    metrics_filename = f"metrics_{base_name.replace('.csv', '.json')}"
    metrics_path = os.path.join(dataset_output_dir, metrics_filename)
    
    with open(metrics_path, 'w') as f:
        json.dump({
            'filename': data_item['filename'],
            'original_file': data_item.get('original_file', data_item['filename']),
            'dataset': data_item['dataset'],
            'num_cycles': int(data_item['num_cycles']),
            'train_end_idx': data_item.get('train_end_idx', 'unknown'),
            'total_length': data_item.get('total_length', 'unknown'),
            'initial_capacity': float(initial_capacity),
            'pinn_model': DATASET_MODEL_MAPPING[data_item['dataset']],
            'metrics': metrics
        }, f, indent=2)
    
    return output_path, metrics_path

def main():
    parser = argparse.ArgumentParser(description='LSTM 결과를 사용한 Dataset별 PINN 추론')
    
    # 입력 경로
    parser.add_argument('--predictions_dir', type=str, 
                       default='C:\\Users\\steve\\Desktop\\새 폴더\\PINN4SOH\\predictions\\TJU_predictions\\',
                       help='LSTM 예측 결과가 있는 디렉토리 (Dataset 폴더들 포함)')
    parser.add_argument('--pinn_model_dir', type=str,
                       default='C:\\Users\\steve\\Desktop\\새 폴더\\PINN4SOH\\pretrained model',
                       help='사전 훈련된 PINN 모델들이 있는 디렉토리')
    
    # 출력 경로
    parser.add_argument('--output_dir', type=str,
                       default='C:\\Users\\steve\\Desktop\\새 폴더\\PINN4SOH\\predictions\\PINN_original_predictions\\',
                       help='PINN 예측 결과를 저장할 디렉토리')
    
    # 모델 설정
    parser.add_argument('--batch_size', type=int, default=512,
                       help='배치 크기 (PINN 모델 설정과 일치)')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='사용할 디바이스')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== 원본 TJU 데이터셋을 사용한 PINN 추론 시작 ===")
    print(f"데이터 디렉토리: {args.data_dir}")
    print(f"학습/예측 비율: {args.train_ratio:.1f}/{1-args.train_ratio:.1f}")
    print(f"특징 컬럼 순서 (16개): {FEATURE_COLUMNS}")
    print(f"Dataset-Model 매핑: {DATASET_MODEL_MAPPING}")
    
    # 1. 원본 TJU 데이터셋 로드
    print(f"\n1. 원본 TJU 데이터셋 로드 중...")
    dataset_data = load_original_tju_dataset(args.data_dir, args.train_ratio)
    
    if not dataset_data:
        print("로드할 Dataset이 없습니다.")
        return
    
    # 2. Dataset별 PINN 추론 수행
    print(f"\n2. Dataset별 PINN 추론 수행...")
    
    all_results = {}
    
    for dataset_name, file_list in dataset_data.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*50}")
        
        try:
            # Dataset에 맞는 PINN 모델 로드
            pinn_model = load_pinn_model_for_dataset(dataset_name, args.pinn_model_dir, args)
            
            dataset_results = []
            
            # 해당 Dataset의 모든 파일 처리
            for i, data_item in enumerate(file_list, 1):
                print(f"\n[{i}/{len(file_list)}] 처리 중: {data_item['filename']}")
                
                try:
                    # PINN으로 예측 (capacity 또는 SOH)
                    pinn_pred = predict_capacity_with_pinn(
                        pinn_model, 
                        data_item['pinn_input']
                    )
                    
                    # 초기 capacity로 true_soh 계산
                    original_df = pd.read_csv(data_item['file'])
                    initial_capacity = original_df['capacity'].iloc[0]
                    capacity_true = data_item['capacity_true']
                    true_soh = capacity_true / initial_capacity
                    
                    # PINN 예측값이 SOH인지 capacity인지 판단 후 변환
                    if np.mean(pinn_pred) > 2.0:  # 평균값이 2.0보다 크면 capacity로 판단
                        print(f"    PINN 예측값이 capacity로 판단됨 (평균: {np.mean(pinn_pred):.3f})")
                        soh_pred = pinn_pred / initial_capacity  # capacity → SOH 변환
                        prediction_type = "capacity"
                    else:  # SOH로 판단
                        print(f"    PINN 예측값이 SOH로 판단됨 (평균: {np.mean(pinn_pred):.3f})")
                        soh_pred = pinn_pred
                        prediction_type = "soh"
                    
                    # SOH 범위 클리핑 (0~1.2 범위로 제한)
                    soh_pred = np.clip(soh_pred, 0.0, 1.2)
                    
                    # 성능 평가 (SOH 기준)
                    metrics = evaluate_predictions(
                        true_soh,  # 실제 SOH
                        soh_pred   # 예측된 SOH
                    )
                    
                    print(f"  예측 완료: {data_item['num_cycles']}개 사이클 ({prediction_type} 예측)")
                    print(f"  성능 - MAE: {metrics['MAE']:.6f}, RMSE: {metrics['RMSE']:.6f}, MAPE: {metrics['MAPE']:.6f}")
                    
                    # 결과 저장 (soh_pred를 전달)
                    output_path, metrics_path = save_predictions(
                        data_item, soh_pred, metrics, args.output_dir
                    )
                    
                    print(f"  저장 완료: {os.path.basename(output_path)}")
                    
                    # Dataset 결과에 추가
                    dataset_results.append({
                        'filename': data_item['filename'],
                        'num_cycles': int(data_item['num_cycles']),
                        'initial_capacity': float(initial_capacity),
                        'prediction_type': prediction_type,
                        'pinn_model': DATASET_MODEL_MAPPING[dataset_name],
                        'metrics': metrics
                    })
                    
                except Exception as e:
                    print(f"  파일 처리 오류: {e}")
                    continue
            
            all_results[dataset_name] = dataset_results
            
            # Dataset별 평균 성능
            if dataset_results:
                print(f"\n{dataset_name} 요약:")
                avg_metrics = {}
                for metric_name in ['MAE', 'MAPE', 'MSE', 'RMSE']:
                    values = [r['metrics'][metric_name] for r in dataset_results]
                    avg_metrics[metric_name] = float(np.mean(values))
                    print(f"  평균 {metric_name}: {avg_metrics[metric_name]:.6f}")
            
        except Exception as e:
            print(f"{dataset_name} 처리 중 오류: {e}")
            continue
    
    # 3. 전체 결과 요약
    print(f"\n{'='*60}")
    print("전체 결과 요약")
    print(f"{'='*60}")
    
    total_files = 0
    all_metrics_combined = []
    
    for dataset_name, dataset_results in all_results.items():
        if dataset_results:
            total_files += len(dataset_results)
            print(f"\n{dataset_name}: {len(dataset_results)}개 파일")
            
            # Dataset별 평균
            avg_metrics = {}
            for metric_name in ['MAE', 'MAPE', 'MSE', 'RMSE']:
                values = [r['metrics'][metric_name] for r in dataset_results]
                avg_metrics[metric_name] = float(np.mean(values))
                all_metrics_combined.extend(values)
                print(f"  평균 {metric_name}: {avg_metrics[metric_name]:.6f}")
    
    # 전체 평균
    if all_results:
        print(f"\n전체 평균 성능:")
        overall_avg = {}
        for metric_name in ['MAE', 'MAPE', 'MSE', 'RMSE']:
            all_values = []
            for dataset_results in all_results.values():
                all_values.extend([r['metrics'][metric_name] for r in dataset_results])
            
            if all_values:
                overall_avg[metric_name] = float(np.mean(all_values))
                print(f"  전체 평균 {metric_name}: {overall_avg[metric_name]:.6f}")
        
        # 전체 요약 저장
        summary = {
            'args': vars(args),
            'dataset_model_mapping': DATASET_MODEL_MAPPING,
            'feature_columns': FEATURE_COLUMNS,
            'total_files': total_files,
            'overall_avg_metrics': overall_avg,
            'dataset_results': all_results
        }
        
        summary_path = os.path.join(args.output_dir, 'pinn_prediction_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n전체 요약 저장: {summary_path}")
        
    else:
        print("처리된 Dataset이 없습니다.")
    
    print(f"\n모든 PINN 예측 결과가 '{os.path.abspath(args.output_dir)}'에 저장되었습니다.")

if __name__ == "__main__":
    main()