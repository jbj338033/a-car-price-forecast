import argparse
from car_price_model import CarPricePredictor
from visualize_results import CarPriceVisualizer
import pandas as pd
import sys
import os

class CarPricePredictionSystem:
    def __init__(self):
        self.predictor = CarPricePredictor()
        self.visualizer = CarPriceVisualizer()
        
    def run_training(self, visualize=True):
        """모델 학습 및 시각화 실행"""
        try:
            print("\n=== 자동차 가격 예측 시스템 시작 ===")
            
            # 모델 학습
            print("\n1. 모델 학습 시작")
            model_metrics = self.predictor.train()
            
            # 랜덤 샘플 평가
            print("\n2. 랜덤 샘플 평가")
            results, samples = self.predictor.evaluate_random_samples(n_samples=5)
            
            # 시각화
            if visualize:
                print("\n3. 결과 시각화")
                self.visualizer.visualize_all()
            
            return model_metrics, results, samples
            
        except Exception as e:
            print(f"\nError in main execution: {str(e)}")
            raise
    
    def predict_new_car(self, car_data):
        """새로운 자동차 데이터에 대한 가격 예측"""
        try:
            print("\n=== 새로운 자동차 가격 예측 ===")
            rf_price, gb_price = self.predictor.predict_price(car_data)
            
            print("\n예측 결과:")
            print(f"RandomForest 예측 가격: £{rf_price[0]:.2f}")
            print(f"GradientBoosting 예측 가격: £{gb_price[0]:.2f}")
            print(f"평균 예측 가격: £{((rf_price[0] + gb_price[0]) / 2):.2f}")
            
            return rf_price[0], gb_price[0]
        
        except Exception as e:
            print(f"\nError in prediction: {str(e)}")
            raise

def parse_arguments():
    parser = argparse.ArgumentParser(description='자동차 가격 예측 시스템')
    parser.add_argument('--no-viz', action='store_true', help='시각화 비활성화')
    parser.add_argument('--predict', action='store_true', help='새로운 자동차 예측 모드')
    parser.add_argument('--input-file', type=str, help='예측할 자동차 데이터 CSV 파일')
    return parser.parse_args()

def main():
    # 명령행 인수 파싱
    args = parse_arguments()
    
    # 시스템 초기화
    system = CarPricePredictionSystem()
    
    try:
        # 모델 학습 및 시각화
        metrics, results, samples = system.run_training(visualize=not args.no_viz)
        
        # 예측 모드가 활성화된 경우
        if args.predict:
            if args.input_file and os.path.exists(args.input_file):
                # CSV 파일에서 예측할 데이터 로드
                new_cars = pd.read_csv(args.input_file)
                print(f"\n{len(new_cars)}개의 새로운 자동차 데이터를 불러왔습니다.")
                
                # 각 자동차에 대해 예측 수행
                predictions = []
                for idx, car in new_cars.iterrows():
                    print(f"\n자동차 {idx + 1} 예측 중...")
                    car_df = pd.DataFrame([car])
                    rf_price, gb_price = system.predict_new_car(car_df)
                    
                    predictions.append({
                        'car_index': idx + 1,
                        'rf_predicted_price': rf_price,
                        'gb_predicted_price': gb_price,
                        'average_predicted_price': (rf_price + gb_price) / 2
                    })
                
                # 예측 결과를 CSV 파일로 저장
                predictions_df = pd.DataFrame(predictions)
                output_file = 'predictions.csv'
                predictions_df.to_csv(output_file, index=False)
                print(f"\n예측 결과가 {output_file}에 저장되었습니다.")
            
            else:
                # 예시 자동차 데이터로 예측
                print("\n예시 자동차 데이터로 예측을 실행합니다.")
                example_car = pd.DataFrame({
                    'year': [2018],
                    'mileage': [21167],
                    'engineSize': [2.0],
                    'model': [1],
                    'transmission': [1],
                    'fuelType': [0],
                    'brand': [3],
                    'tax': [150],
                    'mpg': [50.4]
                })
                
                system.predict_new_car(example_car)
        
        print("\n=== 프로그램 실행 완료 ===")
        
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n오류가 발생했습니다: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()