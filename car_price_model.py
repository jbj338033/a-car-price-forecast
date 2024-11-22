import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

class CarPricePredictor:
    def __init__(self):
        self.rf_model = None
        self.gb_model = None
        self.scaler = None
        self.features = None
        self.processed_df = None
        
    def load_and_combine_data(self):
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
        print("\nStarting data preprocessing...")
        print(f"Initial shape: {df.shape}")
        
        essential_columns = ['model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'engineSize', 'brand']
        if 'tax' in df.columns:
            essential_columns.append('tax')
        if 'mpg' in df.columns:
            essential_columns.append('mpg')
        
        df = df[essential_columns]
        
        if 'tax(£)' in df.columns:
            df = df.drop('tax(£)', axis=1)
        
        if df['price'].dtype == 'object':
            df['price'] = df['price'].str.replace('£', '', regex=False)\
                                    .str.replace(',', '', regex=False)\
                                    .str.replace(' ', '', regex=False)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        if df['mileage'].dtype == 'object':
            df['mileage'] = df['mileage'].str.replace(',', '', regex=False)
            df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
        
        if df['engineSize'].dtype == 'object':
            df['engineSize'] = pd.to_numeric(df['engineSize'], errors='coerce')
        
        le = LabelEncoder()
        categorical_columns = ['model', 'transmission', 'fuelType', 'brand']
        
        self.label_encoders = {}
        for col in categorical_columns:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        if 'tax' in df.columns:
            df['tax'] = df['tax'].fillna(df['tax'].mean())
        if 'mpg' in df.columns:
            df['mpg'] = df['mpg'].fillna(df['mpg'].mean())
        
        df = df.dropna(subset=['model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'engineSize'])
        
        print(f"Final shape after cleaning: {df.shape}")
        print("\nData summary after preprocessing:")
        print(df.describe())
        
        return df

    def prepare_features(self, df):
        print("\nPreparing features...")
        
        features = ['year', 'mileage', 'engineSize', 'model', 'transmission', 'fuelType', 'brand']
        if 'tax' in df.columns:
            features.append('tax')
        if 'mpg' in df.columns:
            features.append('mpg')
        
        print(f"Selected features: {features}")
        
        X = df[features]
        y = df['price']
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features

    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test, features):
        print("\nTraining and comparing models...")
        
        # RandomForest 모델
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)
        rf_pred = self.rf_model.predict(X_test)
        
        # GradientBoosting 모델
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.gb_model.fit(X_train, y_train)
        gb_pred = self.gb_model.predict(X_test)
        
        # 성능 평가
        models = {
            'RandomForest': (self.rf_model, rf_pred),
            'GradientBoosting': (self.gb_model, gb_pred)
        }
        
        model_metrics = {}
        print("\n모델 성능 비교:")
        best_model = None
        best_score = float('-inf')
        
        for model_name, (model, predictions) in models.items():
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            mape = mean_absolute_percentage_error(y_test, predictions)
            
            model_metrics[model_name] = {
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'Predictions': predictions,
                'True_Values': y_test
            }
            
            print(f"\n{model_name} 모델:")
            print(f"RMSE: £{rmse:.2f}")
            print(f"R² Score: {r2:.4f}")
            print(f"MAPE: {mape:.2%}")
            
            if r2 > best_score:
                best_score = r2
                best_model = model_name
        
        print(f"\n최고 성능 모델: {best_model} (R² Score: {best_score:.4f})")
        
        return model_metrics

    def predict_price(self, new_data):
        new_data_scaled = self.scaler.transform(new_data[self.features])
        rf_predicted_price = self.rf_model.predict(new_data_scaled)
        gb_predicted_price = self.gb_model.predict(new_data_scaled)
        
        return rf_predicted_price, gb_predicted_price

    def evaluate_random_samples(self, n_samples=5):
        """랜덤한 샘플을 선택하여 예측 정확도를 평가하는 함수"""
        sample_indices = np.random.choice(len(self.processed_df), n_samples, replace=False)
        samples = self.processed_df.iloc[sample_indices]
        
        print(f"\n{n_samples}개의 랜덤 샘플에 대한 예측 결과:")
        print("\n실제 데이터:")
        print(samples[['year', 'mileage', 'engineSize', 'model', 'transmission', 'fuelType', 'brand', 'price']].to_string())
        
        sample_features = samples[self.features]
        actual_prices = samples['price']
        
        rf_prices, gb_prices = self.predict_price(sample_features)
        
        results = pd.DataFrame({
            'Actual_Price': actual_prices,
            'RF_Predicted': rf_prices,
            'GB_Predicted': gb_prices,
            'RF_Difference': abs(actual_prices - rf_prices),
            'GB_Difference': abs(actual_prices - gb_prices),
            'RF_Error_Percentage': (abs(actual_prices - rf_prices) / actual_prices) * 100,
            'GB_Error_Percentage': (abs(actual_prices - gb_prices) / actual_prices) * 100
        })
        
        print("\n예측 결과 및 오차:")
        print(results.to_string())
        
        print("\n평균 예측 오차:")
        print(f"RandomForest 평균 오차율: {results['RF_Error_Percentage'].mean():.2f}%")
        print(f"GradientBoosting 평균 오차율: {results['GB_Error_Percentage'].mean():.2f}%")
        
        return results, samples

    def train(self):
        """전체 학습 프로세스를 실행하는 함수"""
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
    predictor = CarPricePredictor()
    model_metrics = predictor.train()
    results, samples = predictor.evaluate_random_samples(n_samples=5)