# 자동차 가격 예측 시스템

## 개요

이 프로젝트는 다양한 특성을 기반으로 자동차 가격을 예측하는 머신러닝 시스템입니다. Ridge, Lasso, RandomForest, GradientBoosting 등 여러 모델을 사용하여 앙상블 예측을 수행합니다.

## 설치 방법

1. 필요한 라이브러리 설치:

```bash
pip install -r requirements.txt
```

2. 데이터 준비:
   - data/ 폴더에 자동차 데이터 CSV 파일들을 위치시킵니다.

## 사용 방법

1. 기본 실행 (학습 + 시각화):

```bash
python main.py
```

2. 시각화 없이 실행:

```bash
python main.py --no-viz
```

3. 특정 자동차 데이터 예측:

```bash
python main.py --predict --input-file new_cars.csv
```

4. 랜덤 샘플 수 지정:

```bash
python main.py --n-samples 10
```

## 주요 기능

1. 데이터 전처리 및 특성 엔지니어링
2. 다양한 모델 학습 및 앙상블
3. 하이퍼파라미터 최적화
4. 교차 검증
5. 상세한 성능 평가
6. 다양한 시각화

## 파일 구조

- car_price_model.py: 모델 구현
- visualize_results.py: 시각화 기능
- main.py: 메인 실행 파일
- requirements.txt: 필요 라이브러리
- data/: 데이터 폴더

## 입력 데이터 형식

예측을 위한 CSV 파일은 다음 컬럼을 포함해야 합니다:

- year: 연식
- mileage: 주행거리
- engineSize: 엔진 크기
- model: 모델
- transmission: 변속기 종류
- fuelType: 연료 종류
- brand: 브랜드
- tax: 세금 (선택)
- mpg: 연비 (선택)

## 출력 결과

1. model_predictions.csv: 새로운 데이터 예측 결과
2. random_sample_results.csv: 랜덤 샘플 평가 결과
3. 다양한 시각화 결과

## 참고사항

- 한글 폰트 사용을 위해 Mac OS에서는 AppleGothic 폰트를 사용합니다.
- 모든 수치형 데이터는 자동으로 정규화됩니다.
- 결측치는 평균값으로 대체됩니다.
