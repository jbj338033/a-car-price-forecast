import matplotlib.pyplot as plt
import seaborn as sns
from car_price_model import CarPricePredictor
import pandas as pd
import numpy as np

class CarPriceVisualizer:
    def __init__(self):
        # plt.style.use('seaborn')
        self.predictor = CarPricePredictor()
        
    def visualize_all(self):
        """모든 시각화를 실행하는 함수"""
        # 모델 학습 및 메트릭스 얻기
        model_metrics = self.predictor.train()
        results, samples = self.predictor.evaluate_random_samples(n_samples=5)
        
        # 각종 시각화 실행
        self.plot_prediction_comparison(model_metrics)
        self.plot_error_distribution(model_metrics)
        self.plot_feature_importance()
        self.plot_price_distribution()
        self.plot_correlation_matrix()
        self.plot_sample_predictions(results)
        
        plt.show()
    
    def plot_prediction_comparison(self, model_metrics):
        """실제 가격과 예측 가격 비교 그래프"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(model_metrics['RandomForest']['True_Values'],
                   model_metrics['RandomForest']['Predictions'],
                   alpha=0.5)
        plt.plot([0, max(model_metrics['RandomForest']['True_Values'])],
                [0, max(model_metrics['RandomForest']['True_Values'])],
                'r--')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('RandomForest: Actual vs Predicted')
        
        plt.subplot(1, 2, 2)
        plt.scatter(model_metrics['GradientBoosting']['True_Values'],
                   model_metrics['GradientBoosting']['Predictions'],
                   alpha=0.5)
        plt.plot([0, max(model_metrics['GradientBoosting']['True_Values'])],
                [0, max(model_metrics['GradientBoosting']['True_Values'])],
                'r--')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('GradientBoosting: Actual vs Predicted')
        
        plt.tight_layout()
    
    def plot_error_distribution(self, model_metrics):
        """예측 오차 분포 시각화"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        rf_errors = model_metrics['RandomForest']['True_Values'] - model_metrics['RandomForest']['Predictions']
        sns.histplot(rf_errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title('RandomForest Error Distribution')
        
        plt.subplot(1, 2, 2)
        gb_errors = model_metrics['GradientBoosting']['True_Values'] - model_metrics['GradientBoosting']['Predictions']
        sns.histplot(gb_errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title('GradientBoosting Error Distribution')
        
        plt.tight_layout()
    
    def plot_feature_importance(self):
        """특성 중요도 시각화"""
        rf_importance = pd.DataFrame({
            'feature': self.predictor.features,
            'importance': self.predictor.rf_model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        gb_importance = pd.DataFrame({
            'feature': self.predictor.features,
            'importance': self.predictor.gb_model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 2)
        plt.barh(gb_importance['feature'], gb_importance['importance'])
        plt.xlabel('Importance')
        plt.title('GradientBoosting Feature Importance')
        
        plt.tight_layout()
    
    def plot_price_distribution(self):
        """가격 분포 시각화"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.histplot(data=self.predictor.processed_df, x='price', kde=True)
        plt.xlabel('Price')
        plt.ylabel('Count')
        plt.title('Price Distribution')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=self.predictor.processed_df, x='brand', y='price')
        plt.xticks(rotation=45)
        plt.xlabel('Brand')
        plt.ylabel('Price')
        plt.title('Price Distribution by Brand')
        
        plt.tight_layout()
    
    def plot_correlation_matrix(self):
        """특성 간 상관관계 시각화"""
        numeric_cols = self.predictor.processed_df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.predictor.processed_df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
    
    def plot_sample_predictions(self, results):
        """랜덤 샘플의 예측 결과 시각화"""
        plt.figure(figsize=(12, 5))
        
        # 실제 가격과 예측 가격 비교 그래프
        width = 0.25
        x = np.arange(len(results))
        
        plt.subplot(1, 2, 1)
        plt.bar(x - width, results['Actual_Price'], width, label='Actual')
        plt.bar(x, results['RF_Predicted'], width, label='RandomForest')
        plt.bar(x + width, results['GB_Predicted'], width, label='GradientBoosting')
        plt.xlabel('Sample Index')
        plt.ylabel('Price')
        plt.title('Price Comparison for Random Samples')
        plt.legend()
        
        # 예측 오차율 비교 그래프
        plt.subplot(1, 2, 2)
        plt.bar(x - width/2, results['RF_Error_Percentage'], width, label='RandomForest')
        plt.bar(x + width/2, results['GB_Error_Percentage'], width, label='GradientBoosting')
        plt.xlabel('Sample Index')
        plt.ylabel('Error Percentage')
        plt.title('Prediction Error Percentage')
        plt.legend()
        
        plt.tight_layout()

    def plot_price_trends(self):
        """연도별 가격 추이 시각화"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        year_price = self.predictor.processed_df.groupby('year')['price'].mean()
        plt.plot(year_price.index, year_price.values)
        plt.xlabel('Year')
        plt.ylabel('Average Price')
        plt.title('Average Price by Year')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=self.predictor.processed_df, x='year', y='price')
        plt.xticks(rotation=45)
        plt.xlabel('Year')
        plt.ylabel('Price')
        plt.title('Price Distribution by Year')
        
        plt.tight_layout()

    def plot_mileage_price_relationship(self):
        """주행거리와 가격의 관계 시각화"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(self.predictor.processed_df['mileage'], 
                   self.predictor.processed_df['price'],
                   alpha=0.5)
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.title('Price vs Mileage')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=self.predictor.processed_df, 
                   x='fuelType', 
                   y='price')
        plt.xticks(rotation=45)
        plt.xlabel('Fuel Type')
        plt.ylabel('Price')
        plt.title('Price Distribution by Fuel Type')
        
        plt.tight_layout()

def main():
    visualizer = CarPriceVisualizer()
    visualizer.visualize_all()
    
    # 추가 시각화
    visualizer.plot_price_trends()
    visualizer.plot_mileage_price_relationship()
    
    # 모든 그래프 표시
    plt.show()

if __name__ == "__main__":
    main()
