import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

class RegularizedCarPricePredictor:
    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.ridge_model = None
        self.lasso_model = None
        self.scaler = None
        self.features = None
        self.processed_df = None
        self.label_encoders = {}
        
        # 브랜드 매핑 딕셔너리
        self.brand_mapping = {
            'audi': 'Audi', 'bmw': 'BMW', 'cclass': 'Mercedes C-Class',
            'focus': 'Ford Focus', 'ford': 'Ford', 'hyundi': 'Hyundai',
            'merc': 'Mercedes', 'skoda': 'Skoda', 'toyota': 'Toyota',
            'vauxhall': 'Vauxhall', 'vw': 'Volkswagen'
        }
        
    def load_and_combine_data(self):
        """데이터 로드 및 통합"""
        files = ['audi.csv', 'bmw.csv', 'cclass.csv', 'focus.csv', 'ford.csv', 
                 'hyundi.csv', 'merc.csv', 'skoda.csv', 'toyota.csv', 'vauxhall.csv', 'vw.csv']
        
        dataframes = []
        for file in files:
            try:
                df = pd.read_csv(f'data/{file}')
                print(f"Successfully loaded {file} with {len(df)} rows")
                df['brand'] = file.split('.')[0]
                dataframes.append(df)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"\nTotal combined rows: {len(combined_df)}")
        return combined_df

    def preprocess_data(self, df):
        """데이터 전처리"""
        print("\nStarting data preprocessing...")
        print(f"Initial shape: {df.shape}")
        
        essential_columns = ['model', 'year', 'price', 'transmission', 'mileage', 
                           'fuelType', 'engineSize', 'brand']
        if 'tax' in df.columns:
            essential_columns.append('tax')
        if 'mpg' in df.columns:
            essential_columns.append('mpg')
        
        df = df[essential_columns]
        
        # 가격 전처리
        if df['price'].dtype == 'object':
            df['price'] = df['price'].str.replace('£', '', regex=False)\
                                    .str.replace(',', '', regex=False)\
                                    .str.replace(' ', '', regex=False)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # 주행거리 전처리
        if df['mileage'].dtype == 'object':
            df['mileage'] = df['mileage'].str.replace(',', '', regex=False)
            df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
        
        # 엔진 크기 전처리
        if df['engineSize'].dtype == 'object':
            df['engineSize'] = pd.to_numeric(df['engineSize'], errors='coerce')
        
        # 범주형 변수 인코딩
        categorical_columns = ['model', 'transmission', 'fuelType', 'brand']
        for col in categorical_columns:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        # 결측치 처리
        if 'tax' in df.columns:
            df['tax'] = df['tax'].fillna(df['tax'].mean())
        if 'mpg' in df.columns:
            df['mpg'] = df['mpg'].fillna(df['mpg'].mean())
        
        # 필수 컬럼의 결측치가 있는 행 제거
        df = df.dropna(subset=['model', 'year', 'price', 'transmission', 
                              'mileage', 'fuelType', 'engineSize'])
        
        print(f"Final shape after cleaning: {df.shape}")
        print("\nData summary after preprocessing:")
        print(df.describe())
        
        return df

    def prepare_features(self, df):
        """특성 준비 및 스케일링"""
        print("\nPreparing features...")
        
        features = ['year', 'mileage', 'engineSize', 'model', 'transmission', 
                   'fuelType', 'brand']
        if 'tax' in df.columns:
            features.append('tax')
        if 'mpg' in df.columns:
            features.append('mpg')
        
        print(f"Selected features: {features}")
        
        X = df[features]
        y = df['price']
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                           random_state=42)
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features

    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test, features):
        """모델 학습 및 평가"""
        print("\n모델 학습 및 평가 중...")
        
        # 하이퍼파라미터 그리드 정의
        ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
        lasso_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        gb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1]
        }

        # 모델 정의
        models = {
            'Ridge': (Ridge(), ridge_params),
            'Lasso': (Lasso(), lasso_params),
            'RandomForest': (RandomForestRegressor(random_state=42), rf_params),
            'GradientBoosting': (GradientBoostingRegressor(random_state=42), gb_params)
        }

        best_models = {}
        model_metrics = {}
        
        # 각 모델 학습 및 평가
        for model_name, (model, params) in models.items():
            print(f"\n{model_name} 모델 튜닝 중...")
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=5,
                scoring='r2',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_models[model_name] = grid_search.best_estimator_
            
            # 예측 및 평가
            predictions = grid_search.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            mape = mean_absolute_percentage_error(y_test, predictions)
            
            # 교차 검증 점수
            cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, 
                                      cv=5, scoring='r2')
            
            model_metrics[model_name] = {
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'CV_Scores': cv_scores,
                'Best_Params': grid_search.best_params_,
                'Predictions': predictions,
                'True_Values': y_test
            }
            
            print(f"\n{model_name} 모델 결과:")
            print(f"최적 파라미터: {grid_search.best_params_}")
            print(f"RMSE: £{rmse:.2f}")
            print(f"R² Score: {r2:.4f}")
            print(f"MAPE: {mape:.2%}")
            print(f"교차 검증 R² Scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # 최적 모델 저장
        self.ridge_model = best_models['Ridge']
        self.lasso_model = best_models['Lasso']
        self.rf_model = best_models['RandomForest']
        self.gb_model = best_models['GradientBoosting']

        return model_metrics

    def predict_price(self, new_data):
        """가격 예측 수행"""
        new_data_scaled = self.scaler.transform(new_data[self.features])
        
        # 각 모델의 예측값
        ridge_pred = self.ridge_model.predict(new_data_scaled)
        lasso_pred = self.lasso_model.predict(new_data_scaled)
        rf_pred = self.rf_model.predict(new_data_scaled)
        gb_pred = self.gb_model.predict(new_data_scaled)
        
        # 테스트 세트 성능 기반 가중치 계산
        weights = {
            'Ridge': 0.7170,      # Ridge의 R² Score
            'Lasso': 0.7170,      # Lasso의 R² Score
            'RandomForest': 0.9605,  # RandomForest의 R² Score
            'GradientBoosting': 0.9526   # GradientBoosting의 R² Score
        }
        
        # 가중치 정규화
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # 가중 평균 예측값
        ensemble_pred = (
            normalized_weights['Ridge'] * ridge_pred +
            normalized_weights['Lasso'] * lasso_pred +
            normalized_weights['RandomForest'] * rf_pred +
            normalized_weights['GradientBoosting'] * gb_pred
        )
        
        return ridge_pred, lasso_pred, rf_pred, gb_pred, ensemble_pred

    def evaluate_random_samples(self, n_samples=5):
        """랜덤 샘플 평가"""
        sample_indices = np.random.choice(len(self.processed_df), n_samples, replace=False)
        samples = self.processed_df.iloc[sample_indices]
        
        print(f"\n{n_samples}개의 랜덤 샘플에 대한 예측 결과:")
        print("\n실제 데이터:")
        print(samples[['year', 'mileage', 'engineSize', 'model', 'transmission', 
                      'fuelType', 'brand', 'price']].to_string())
        
        sample_features = samples[self.features]
        actual_prices = samples['price']
        
        ridge_prices, lasso_prices, rf_prices, gb_prices, ensemble_prices = self.predict_price(sample_features)
        
        results = pd.DataFrame({
            'Actual_Price': actual_prices,
            'Ridge_Predicted': ridge_prices,
            'Lasso_Predicted': lasso_prices,
            'RF_Predicted': rf_prices,
            'GB_Predicted': gb_prices,
            'Ensemble_Predicted': ensemble_prices,
            'Ridge_Error_%': (abs(actual_prices - ridge_prices) / actual_prices) * 100,
            'Lasso_Error_%': (abs(actual_prices - lasso_prices) / actual_prices) * 100,
            'RF_Error_%': (abs(actual_prices - rf_prices) / actual_prices) * 100,
            'GB_Error_%': (abs(actual_prices - gb_prices) / actual_prices) * 100,
            'Ensemble_Error_%': (abs(actual_prices - ensemble_prices) / actual_prices) * 100
        })
        
        print("\n예측 결과 및 오차율:")
        print(results.to_string())
        
        print("\n모델별 평균 오차율:")
        error_columns = [col for col in results.columns if 'Error' in col]
        mean_errors = results[error_columns].mean()
        for model, error in mean_errors.items():
            print(f"{model}: {error:.2f}%")
        
        return results, samples

    def train(self):
        """전체 학습 프로세스"""
        try:
            print("데이터 로드 중...")
            combined_df = self.load_and_combine_data()
            
            print("데이터 전처리 중...")
            self.processed_df = self.preprocess_data(combined_df)
            
            print("특성 준비 중...")
            X_train, X_test, y_train, y_test, self.scaler, self.features = self.prepare_features(self.processed_df)
            
            print("모델 학습 및 비교 중...")
            model_metrics = self.train_and_evaluate_models(X_train, X_test, y_train, y_test, self.features)
            
            return model_metrics
            
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            raise

if __name__ == "__main__":
    predictor = RegularizedCarPricePredictor()
    model_metrics = predictor.train()
    results, samples = predictor.evaluate_random_samples(n_samples=5)