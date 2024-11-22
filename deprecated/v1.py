import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터 로드 및 통합 (이전과 동일)
def load_and_combine_data():
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

# 2. 데이터 전처리 (이전과 동일)
def preprocess_data(df):
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
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    
    if 'tax' in df.columns:
        df['tax'] = df['tax'].fillna(df['tax'].mean())
    if 'mpg' in df.columns:
        df['mpg'] = df['mpg'].fillna(df['mpg'].mean())
    
    df = df.dropna(subset=['model', 'year', 'price', 'transmission', 'mileage', 'fuelType', 'engineSize'])
    
    print(f"Final shape after cleaning: {df.shape}")
    print("\nData summary after preprocessing:")
    print(df.describe())
    
    return df

# 3. 특성 선택 및 스케일링 (이전과 동일)
def prepare_features(df):
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

# 4. 모델 학습 및 평가 (수정됨)
def train_and_evaluate_model(X_train, X_test, y_train, y_test, features):
    print("\nTraining model...")
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n모델 성능 평가:")
    print(f"RMSE: £{rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n특성 중요도:")
    print(feature_importance)
    
    return model

# 5. 새로운 데이터에 대한 예측 함수 (이전과 동일)
def predict_price(model, scaler, features, new_data):
    new_data_scaled = scaler.transform(new_data[features])
    predicted_price = model.predict(new_data_scaled)
    return predicted_price

# 메인 실행 함수 (수정됨)
def main():
    try:
        print("데이터 로드 중...")
        combined_df = load_and_combine_data()
        
        print("데이터 전처리 중...")
        processed_df = preprocess_data(combined_df)
        
        print("특성 준비 중...")
        X_train, X_test, y_train, y_test, scaler, features = prepare_features(processed_df)
        
        print("모델 학습 중...")
        model = train_and_evaluate_model(X_train, X_test, y_train, y_test, features)
        
        return model, scaler, features, processed_df
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    model, scaler, features, processed_df = main()
    
    # 예측 예시
    print("\n예측 테스트:")
    new_car = pd.DataFrame({
        'year': [2018],
        'mileage': [21167],
        'engineSize': [2.0],
        'model': [1],  # Focus
        'transmission': [1],  # Manual
        'fuelType': [0],  # Diesel
        'brand': [3],  # focus.csv
        'tax': [150],  # 평균적인 값 사용
        'mpg': [50.4]  # 평균적인 값 사용
    })

    predicted_price = predict_price(model, scaler, features, new_car)
    print(f"예측 가격: £{predicted_price[0]:.2f}")