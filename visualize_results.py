import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from regularized_car_price_model import RegularizedCarPricePredictor
import platform
from sklearn.model_selection import learning_curve

class CarPriceVisualizer:
    def __init__(self):
        # 한글 폰트 설정
        if platform.system() == 'Darwin':  # macOS
            plt.rc('font', family='AppleGothic')
        plt.rc('axes', unicode_minus=False)
        
        self.predictor = RegularizedCarPricePredictor()
        self.set_style()
        
    def set_style(self):
        """시각화 스타일 설정"""
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
    
    def visualize_all(self):
        """모든 시각화 수행"""
        print("모델 학습 및 데이터 준비 중...")
        self.model_metrics = self.predictor.train()
        self.results, self.samples = self.predictor.evaluate_random_samples(n_samples=5)
        
        print("\n다양한 시각화 생성 중...")
        self.plot_model_comparison()
        self.plot_feature_importance()
        self.plot_residuals()
        self.plot_learning_curves()
        self.plot_price_analysis()
        self.plot_correlation_analysis()
        self.plot_error_analysis()
        
        plt.show()
    
    def plot_model_comparison(self):
        """모델 성능 비교 시각화"""
        plt.figure(figsize=(15, 10))
        
        # 1. R² Score 비교
        plt.subplot(2, 2, 1)
        r2_scores = {name: metrics['R2'] 
                    for name, metrics in self.model_metrics.items()}
        plt.bar(r2_scores.keys(), r2_scores.values())
        plt.title('모델별 R² Score 비교')
        plt.xticks(rotation=45)
        plt.ylabel('R² Score')
        
        # 2. RMSE 비교
        plt.subplot(2, 2, 2)
        rmse_scores = {name: metrics['RMSE'] 
                      for name, metrics in self.model_metrics.items()}
        plt.bar(rmse_scores.keys(), rmse_scores.values())
        plt.title('모델별 RMSE 비교')
        plt.xticks(rotation=45)
        plt.ylabel('RMSE (£)')
        
        # 3. MAPE 비교
        plt.subplot(2, 2, 3)
        mape_scores = {name: metrics['MAPE'] 
                      for name, metrics in self.model_metrics.items()}
        plt.bar(mape_scores.keys(), mape_scores.values())
        plt.title('모델별 MAPE 비교')
        plt.xticks(rotation=45)
        plt.ylabel('MAPE')
        
        # 4. 교차 검증 점수 분포
        plt.subplot(2, 2, 4)
        cv_scores = {name: metrics['CV_Scores'] 
                    for name, metrics in self.model_metrics.items()}
        plt.boxplot([scores for scores in cv_scores.values()], 
                   labels=cv_scores.keys())
        plt.title('교차 검증 점수 분포')
        plt.xticks(rotation=45)
        plt.ylabel('R² Score')
        
        plt.tight_layout()
    
    def plot_feature_importance(self):
        """특성 중요도 시각화"""
        plt.figure(figsize=(15, 6))
        
        # RandomForest 특성 중요도
        plt.subplot(1, 2, 1)
        rf_importance = pd.DataFrame({
            'feature': self.predictor.features,
            'importance': self.predictor.rf_model.feature_importances_
        }).sort_values('importance', ascending=True)
        plt.barh(rf_importance['feature'], rf_importance['importance'])
        plt.title('RandomForest 특성 중요도')
        
        # GradientBoosting 특성 중요도
        plt.subplot(1, 2, 2)
        gb_importance = pd.DataFrame({
            'feature': self.predictor.features,
            'importance': self.predictor.gb_model.feature_importances_
        }).sort_values('importance', ascending=True)
        plt.barh(gb_importance['feature'], gb_importance['importance'])
        plt.title('GradientBoosting 특성 중요도')
        
        plt.tight_layout()
    
    def plot_residuals(self):
        """잔차 분석 시각화"""
        plt.figure(figsize=(15, 10))
        
        for idx, (name, metrics) in enumerate(self.model_metrics.items(), 1):
            plt.subplot(2, 2, idx)
            residuals = metrics['True_Values'] - metrics['Predictions']
            sns.histplot(residuals, kde=True)
            plt.title(f'{name} 잔차 분포')
            plt.xlabel('잔차')
            plt.ylabel('빈도')
        
        plt.tight_layout()
    
    def plot_learning_curves(self):
        """학습 곡선 시각화"""
        plt.figure(figsize=(15, 10))
        
        for idx, (name, model) in enumerate([
            ('Ridge', self.predictor.ridge_model),
            ('Lasso', self.predictor.lasso_model),
            ('RandomForest', self.predictor.rf_model),
            ('GradientBoosting', self.predictor.gb_model)
        ], 1):
            plt.subplot(2, 2, idx)
            train_sizes, train_scores, test_scores = learning_curve(
                model, 
                self.predictor.processed_df[self.predictor.features],
                self.predictor.processed_df['price'],
                train_sizes=np.linspace(0.1, 1.0, 10),
                cv=5,
                scoring='r2'
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            plt.plot(train_sizes, train_mean, label='학습 점수')
            plt.plot(train_sizes, test_mean, label='검증 점수')
            plt.fill_between(train_sizes, train_mean - train_std,
                           train_mean + train_std, alpha=0.1)
            plt.fill_between(train_sizes, test_mean - test_std,
                           test_mean + test_std, alpha=0.1)
            plt.title(f'{name} 학습 곡선')
            plt.xlabel('학습 데이터 크기')
            plt.ylabel('R² Score')
            plt.legend(loc='best')
        
        plt.tight_layout()
    
    def plot_price_analysis(self):
        """가격 분석 시각화"""
        df = self.predictor.processed_df
        plt.figure(figsize=(15, 10))
        
        # 1. 가격 분포
        plt.subplot(2, 2, 1)
        sns.histplot(data=df, x='price', kde=True)
        plt.title('가격 분포')
        plt.xlabel('가격 (£)')
        
        # 2. 브랜드별 가격 분포
        plt.subplot(2, 2, 2)
        df['brand_name'] = df['brand'].map(
            {idx: name for idx, name in enumerate(self.predictor.brand_mapping.values())})
        sns.boxplot(data=df, x='brand_name', y='price')
        plt.xticks(rotation=45)
        plt.title('브랜드별 가격 분포')
        
        # 3. 연도별 가격 추이
        plt.subplot(2, 2, 3)
        sns.scatterplot(data=df, x='year', y='price', alpha=0.5)
        plt.title('연도별 가격 추이')
        
        # 4. 주행거리와 가격의 관계
        plt.subplot(2, 2, 4)
        sns.scatterplot(data=df, x='mileage', y='price', alpha=0.5)
        plt.title('주행거리와 가격의 관계')
        
        plt.tight_layout()
    
    def plot_correlation_analysis(self):
        """상관관계 분석 시각화"""
        df = self.predictor.processed_df
        
        # 숫자형 변수만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('변수 간 상관관계')
        plt.tight_layout()
    
    def plot_error_analysis(self):
        """예측 오차 분석 시각화"""
        plt.figure(figsize=(15, 10))
        
        # 1. 모델별 예측 오차율 비교
        plt.subplot(2, 2, 1)
        error_cols = [col for col in self.results.columns if 'Error' in col]
        error_means = self.results[error_cols].mean()
        plt.bar(error_means.index, error_means.values)
        plt.title('모델별 평균 오차율')
        plt.xticks(rotation=45)
        plt.ylabel('평균 오차율 (%)')
        
        # 2. 실제가격 대비 예측가격 산점도
        plt.subplot(2, 2, 2)
        actual = self.results['Actual_Price']
        ensemble = self.results['Ensemble_Predicted']
        plt.scatter(actual, ensemble, alpha=0.5)
        plt.plot([actual.min(), actual.max()], 
                [actual.min(), actual.max()], 'r--')
        plt.title('실제가격 vs 앙상블 예측가격')
        plt.xlabel('실제 가격 (£)')
        plt.ylabel('예측 가격 (£)')
        
        # 3. 오차율 분포
        plt.subplot(2, 2, 3)
        sns.histplot(self.results['Ensemble_Error_%'], kde=True)
        plt.title('앙상블 모델 오차율 분포')
        plt.xlabel('오차율 (%)')
        
        # 4. 가격 구간별 오차율
        plt.subplot(2, 2, 4)
        self.results['Price_Bin'] = pd.qcut(self.results['Actual_Price'], 
                                          q=5, labels=['매우 낮음', '낮음', '중간', '높음', '매우 높음'])
        sns.boxplot(data=self.results, x='Price_Bin', y='Ensemble_Error_%')
        plt.title('가격 구간별 오차율 분포')
        plt.xticks(rotation=45)
        plt.xlabel('가격 구간')
        plt.ylabel('오차율 (%)')
        
        plt.tight_layout()

def main():
    visualizer = CarPriceVisualizer()
    visualizer.visualize_all()

if __name__ == "__main__":
    main()