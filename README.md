# Car Price Prediction Project

## 개요

이 프로젝트는 자동차 데이터를 기반으로 가격을 예측하는 머신러닝 모델을 구현합니다. RandomForest와 GradientBoosting 두 가지 모델을 사용하여 예측을 수행하고, 다양한 시각화를 통해 결과를 분석합니다.

## 프로젝트 구조

```
project/
│
├── data/                 # 데이터 파일들
│   ├── audi.csv
│   ├── bmw.csv
│   └── ...
│
├── car_price_model.py    # 모델 학습 및 예측 코드
├── visualize_results.py  # 결과 시각화 코드
└── README.md            # 프로젝트 설명
```

## 필요한 라이브러리

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 사용 방법

1. 데이터 준비:

   - `data/` 폴더에 자동차 데이터 CSV 파일들을 위치시킵니다.

2. 모델 학습:

```python
from car_price_model import CarPricePredictor

predictor = CarPricePredictor()
model_metrics = predictor.train()
results, samples = predictor.evaluate_random_samples(n_samples=5)
```

3. 결과 시각화:

```python
from visualize_results import CarPriceVisualizer

visualizer = CarPriceVisualizer()
visualizer.visualize_all()
```

## 주요 기능

### CarPricePredictor 클래스

- 데이터 로드 및 전처리
- 모델 학습 및 평가
- 랜덤 샘플에 대한 예측 성능 평가

### CarPriceVisualizer 클래스

- 실제 가격과 예측 가격 비교
- 예측 오차 분포 시각화
- 특성 중요도 시각화
- 가격 분포 시각화
- 상관관계 매트릭스
- 샘플 예측 결과 시각화
- 연도별 가격 추이
- 주행거리와 가격의 관계

## 시각화 결과

모델은 다음과 같은 다양한 시각화를 제공합니다:

1. 모델 예측 성능 비교
2. 특성 중요도 분석
3. 가격 분포 분석
4. 브랜드별 가격 분포
5. 연도별 가격 추이
6. 주행거리와 가격의 관계
7. 예측 오차 분석

## 참고사항

- 모든 수치형 데이터는 StandardScaler를 사용하여 정규화됩니다.
- 범주형 데이터는 LabelEncoder를 사용하여 인코딩됩니다.
- 결측치는 평균값으로 대체됩니다.
