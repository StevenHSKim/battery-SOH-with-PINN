# E2E framework with LSTM & PINN for battery SOH prediction
배터리 연구/개발 단계에서 내구시험에 상당한 시간이 소요되는 문제가 존재한다. 
본 연구에서는 배터리 내구 검사 시간 단축을 위한 LSTM&PINN 기반 배터리 SOH 예측 프레임워크를 제안한다.

# 1. System requirements
python version: 3.7.10

|    Package     | Version  |
|:--------------:|:--------:|
|     torch      |  1.7.1   |
|    sklearn     |  0.24.2  |
|     numpy      |  1.20.3  |
|     pandas     |  1.3.5   |
|   matplotlib   |  3.3.4   |
|  scienceplots  |          |



# 2. Installation guide
If you are not familiar with Python and Pytorch framework, 
you can install Anaconda first and use Anaconda to quickly configure the environment.
## 2.1 Create environment
```angular2html
conda create -n new_environment python=3.7.10
```



## 2.2 Activate environment
```angular2html
conda activate new_environment
```

## 2.3 Install dependencies
```angular2html
conda install pytorch=1.7.1
conda install scikit-learn=0.24.2 numpy=1.20.3 pandas=1.3.5 matplotlib=3.3.4
pip install scienceplots      # for beautiful plots
```

