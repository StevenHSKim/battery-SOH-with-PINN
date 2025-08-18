# PINN4SOH
This code is for this paper: [Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis](https://www.nature.com/articles/s41467-024-48779-z)

> [!IMPORTANT]
Summary of articles using the XJTU Battery Dataset:
https://github.com/wang-fujin/XJTU-Battery-Dataset-Papers-Summary

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

