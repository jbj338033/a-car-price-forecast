from regularized_car_price_model import RegularizedCarPricePredictor
from visualize_results import CarPriceVisualizer
import pandas as pd
import numpy as np
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='자동차 가격 예측 시스템')
    parser.add_argument('--no-viz', action='store_true', 
                       help='시각화 비활성화')
    parser.add_argument('--predict', action='store_true',
                       help='새로운 자동차 예측 모드')
    parser.add_argument('--input-file', type=str,
                       help='예측할 자동차 데이터 CSV 파일')
    parser.add_argument('--n-samples', type=int, default=5,
                       help='랜덤 샘플 평가 개수')
    return parser.parse_args()

def save_results(results, filename='prediction_results.csv'):
    """결과를 CSV 파일로 저장"""
    results.to_csv(filename, index=False)
    print(f"\n결과가 {filename}에 저장되었습니다.")

def main():
    args = parse_arguments()
    
    try:
        # 모델 및 시각화 객체 초기화
        predictor = RegularizedCarPricePredictor()
        
        # 모델 학습
        print("\n=== 모델 학습 시작 ===")
        model_metrics = predictor.train()
        
        # 시각화
        if not args.no_viz:
            print("\n=== 결과 시각화 시작 ===")
            visualizer = CarPriceVisualizer()
            visualizer.visualize_all()
        
        # 예측 모드
        if args.predict:
            print("\n=== 예측 모드 실행 ===")
            if args.input_file and os.path.exists(args.input_file):
                # CSV 파일에서 데이터 로드
                new_cars = pd.read_csv(args.input_file)
                print(f"\n{len(new_cars)}개의 새로운 자동차 데이터를 불러왔습니다.")
                
                results_list = []
                for idx, car in new_cars.iterrows():
                    print(f"\n자동차 {idx + 1} 예측 중...")
                    car_df = pd.DataFrame([car])
                    ridge_pred, lasso_pred, rf_pred, gb_pred, ensemble_pred = predictor.predict_price(car_df)
                    
                    results_list.append({
                        'Car_Index': idx + 1,
                        'Ridge_Price': ridge_pred[0],
                        'Lasso_Price': lasso_pred[0],
                        'RandomForest_Price': rf_pred[0],
                        'GradientBoosting_Price': gb_pred[0],
                        'Ensemble_Price': ensemble_pred[0]
                    })
                
                # 결과를 DataFrame으로 변환하고 저장
                results_df = pd.DataFrame(results_list)
                save_results(results_df, 'model_predictions.csv')
                
            else:
                # 예시 자동차 데이터로 예측
                print("\n예시 자동차 데이터로 예측을 실행합니다.")
                example_car = pd.DataFrame({
                    'year': [2018],
                    'mileage': [21167],
                    'engineSize': [2.0],
                    'model': [1],  # 예: Focus
                    'transmission': [1],  # 예: Manual
                    'fuelType': [0],  # 예: Petrol
                    'brand': [3],  # 예: Ford
                    'tax': [150],
                    'mpg': [50.4]
                })
                
                ridge_pred, lasso_pred, rf_pred, gb_pred, ensemble_pred = predictor.predict_price(example_car)
                
                print("\n예측 결과:")
                print(f"Ridge 예측 가격: £{ridge_pred[0]:.2f}")
                print(f"Lasso 예측 가격: £{lasso_pred[0]:.2f}")
                print(f"RandomForest 예측 가격: £{rf_pred[0]:.2f}")
                print(f"GradientBoosting 예측 가격: £{gb_pred[0]:.2f}")
                print(f"앙상블 예측 가격: £{ensemble_pred[0]:.2f}")
        
        # 랜덤 샘플 평가
        print(f"\n=== {args.n_samples}개의 랜덤 샘플 평가 ===")
        results, samples = predictor.evaluate_random_samples(n_samples=args.n_samples)
        save_results(results, 'random_sample_results.csv')
        
        print("\n=== 프로그램 실행 완료 ===")
        
    except Exception as e:
        print(f"\n오류가 발생했습니다: {str(e)}")
        raise

if __name__ == "__main__":
    main()