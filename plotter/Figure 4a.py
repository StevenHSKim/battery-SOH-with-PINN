'''
PINN 예측 결과 시각화
Original structure maintained
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# LaTeX 비활성화
plt.rcParams['text.usetex'] = False

# scienceplots 스타일 적용 (LaTeX 없이)
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except:
    print("scienceplots 없이 기본 스타일 사용")

# 3행4열 subplot 생성 (원본과 동일)
fig, axs = plt.subplots(3, 4, figsize=(8, 6), dpi=150)
count = 0

# 색상 설정 (원본과 동일)
color_list = [
    '#74AED4',
    '#7BDFF2',
    '#FBDD85',
    '#F46F43',
    '#CF3D3E'
]
colors = plt.cm.colors.LinearSegmentedColormap.from_list(
    'custom_cmap', color_list, N=256
)

# PINN 결과 디렉토리 (현재 환경에 맞게 수정)
results_dir = "C:\\Users\\steve\\Desktop\\새 폴더\\PINN4SOH\\predictions\\PINN_predictions"

# CSV 파일들 찾기 (Dataset 폴더들에서 재귀적으로 검색)
csv_pattern = os.path.join(results_dir, "**", "pinn_*.csv")
csv_files = glob.glob(csv_pattern, recursive=True)

print(f"발견된 CSV 파일: {len(csv_files)}개")

# 각 CSV 파일 처리 (원본 구조와 동일한 루프)
for csv_file in csv_files:
    if count >= 11:  # 최대 11개까지 (원본과 동일)
        break
        
    try:
        # 데이터 읽기
        df = pd.read_csv(csv_file)
        
        # 필요한 컬럼 확인 (현재 환경의 컬럼명에 맞게 수정)
        if 'soh_pred' not in df.columns or 'true_soh' not in df.columns:
            print(f"필요한 컬럼이 없습니다: {csv_file}")
            continue
            
        # 예측 구간만 필터링
        predict_mask = df['data_type'] == 'predict'
        predict_df = df[predict_mask]
        
        if len(predict_df) == 0:
            print(f"예측 구간이 없습니다: {csv_file}")
            continue
            
        # 데이터 추출 (이미 0~1 범위)
        pred_label = predict_df['soh_pred'].values
        true_label = predict_df['true_soh'].values
        
        # NaN 값 제거
        valid_mask = ~(np.isnan(pred_label) | np.isnan(true_label))
        pred_label = pred_label[valid_mask]
        true_label = true_label[valid_mask]
        
        if len(pred_label) == 0:
            print(f"유효한 데이터가 없습니다: {csv_file}")
            continue
        
        # 파일명에서 배터리 이름 추출
        filename = os.path.basename(csv_file)
        battery_name = filename.replace('pinn_', '').replace('.csv', '')
        title = f'{battery_name}'
        
    except Exception as e:
        print(f"파일 로드 실패: {csv_file} - {e}")
        continue
    
    # 오차 계산 (원본과 동일)
    error = np.abs(pred_label - true_label)
    vmin, vmax = error.min(), error.max()
    
    # 축 범위 설정 (데이터 범위에 맞게 조정)
    data_min = max(0.6, min(true_label.min(), pred_label.min()) - 0.05)
    data_max = min(1.1, max(true_label.max(), pred_label.max()) + 0.05)
    lims = [data_min, data_max]
    
    # subplot 위치 계산 (원본과 동일)
    col = count % 4
    row = count // 4
    print(f"{battery_name}, count={count}, row={row}, col={col}")
    ax = axs[row, col]
    
    # 산점도 그리기 (원본과 동일)
    ax.scatter(true_label, pred_label, c=error, cmap=colors, s=3, alpha=0.7, vmin=0, vmax=0.1)
    
    # 대각선 그리기 (원본과 동일)
    ax.plot([data_min, data_max], [data_min, data_max], '--', c='#ff4d4e', alpha=1, linewidth=1)
    ax.set_aspect('equal')
    ax.set_xlabel('True SOH')
    ax.set_ylabel('Prediction')
    
    # 틱 설정 (데이터 범위에 맞게)
    tick_range = np.arange(0.6, 1.15, 0.1)
    visible_ticks = tick_range[(tick_range >= data_min) & (tick_range <= data_max)]
    ax.set_xticks(visible_ticks)
    ax.set_yticks(visible_ticks)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # 제목 설정 (원본과 동일)
    ax.set_title(title)
    
    count += 1

# 사용하지 않는 subplot 숨기기
for i in range(count, 12):
    row = i // 4
    col = i % 4
    if row < 3 and col < 4:
        axs[row, col].axis('off')

# 마지막 subplot에 colorbar 추가 (원본과 동일)
fig.colorbar(plt.cm.ScalarMappable(cmap=colors, norm=plt.Normalize(vmin=0, vmax=0.1)),
             ax=axs[2, 3],
             label='Absolute error',
             fraction=0.46, pad=0.4)

# 마지막 subplot 축 끄기 (원본과 동일)
axs[2, 3].axis('off')

# 레이아웃 조정 및 표시 (원본과 동일)
plt.tight_layout()

# 저장 (옵션)
save_path = os.path.join(results_dir, 'pinn_estimation_results.png')
plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
print(f"그래프 저장: {save_path}")

plt.show()