import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from car_price_model import CarPricePredictor
import matplotlib.font_manager as fm
import platform

class EnhancedCarPriceVisualizer:
    def __init__(self):
        # 한글 폰트 설정
        if platform.system() == 'Darwin':  # macOS
            plt.rc('font', family='AppleGothic')
        plt.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지
        
        # plt.style.use('seaborn')
        self.predictor = CarPricePredictor()
        self.set_style()
        
    def set_style(self):
        """시각화 스타일 설정"""
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        
    def visualize_all(self):
        """모든 시각화 실행"""
        print("모델 학습 및 데이터 준비 중...")
        self.model_metrics = self.predictor.train()
        self.results, self.samples = self.predictor.evaluate_random_samples(n_samples=5)
        
        print("\n다양한 시각화 생성 중...")
        self.create_basic_analysis_plots()
        self.create_advanced_analysis_plots()
        self.create_model_performance_plots()
        self.create_relationship_plots()
        self.create_distribution_plots()
        
        plt.show()
    
    def create_basic_analysis_plots(self):
        """기본 데이터 분석 시각화"""
        # 1. 브랜드별 평균 가격
        plt.figure(figsize=(15, 6))
        brand_avg = self.predictor.processed_df.groupby('brand')['price'].mean().sort_values(ascending=False)
        sns.barplot(x=brand_avg.index, y=brand_avg.values)
        plt.title('브랜드별 평균 가격')
        plt.xticks(rotation=45)
        plt.xlabel('브랜드')
        plt.ylabel('평균 가격 (£)')
        plt.tight_layout()
        
        # 2. 연도별 가격 트렌드
        plt.figure(figsize=(15, 6))
        year_avg = self.predictor.processed_df.groupby('year')['price'].agg(['mean', 'min', 'max'])
        plt.fill_between(year_avg.index, year_avg['min'], year_avg['max'], alpha=0.2)
        plt.plot(year_avg.index, year_avg['mean'], marker='o')
        plt.title('연도별 가격 추이')
        plt.xlabel('연도')
        plt.ylabel('가격 (£)')
        plt.legend(['평균 가격', '가격 범위'])
        plt.tight_layout()
        
        # 3. 엔진 크기별 분포
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(data=self.predictor.processed_df, x='engineSize', bins=30)
        plt.title('엔진 크기 분포')
        plt.xlabel('엔진 크기')
        plt.ylabel('빈도')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=self.predictor.processed_df, x='engineSize', y='price')
        plt.title('엔진 크기별 가격 분포')
        plt.xlabel('엔진 크기')
        plt.ylabel('가격 (£)')
        plt.tight_layout()
    
    def create_advanced_analysis_plots(self):
        """고급 데이터 분석 시각화"""
        # 1. 주행거리와 가격의 관계 (산점도 + 회귀선)
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        sns.regplot(data=self.predictor.processed_df, x='mileage', y='price', scatter_kws={'alpha':0.5})
        plt.title('주행거리와 가격의 관계')
        plt.xlabel('주행거리')
        plt.ylabel('가격 (£)')
        
        plt.subplot(1, 2, 2)
        sns.regplot(data=self.predictor.processed_df, x='year', y='price', scatter_kws={'alpha':0.5})
        plt.title('연식과 가격의 관계')
        plt.xlabel('연도')
        plt.ylabel('가격 (£)')
        plt.tight_layout()
        
        # 2. 연료 타입별 가격 분포
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        sns.violinplot(data=self.predictor.processed_df, x='fuelType', y='price')
        plt.title('연료 타입별 가격 분포')
        plt.xlabel('연료 타입')
        plt.ylabel('가격 (£)')
        
        plt.subplot(1, 2, 2)
        sns.boxenplot(data=self.predictor.processed_df, x='transmission', y='price')
        plt.title('변속기 종류별 가격 분포')
        plt.xlabel('변속기 종류')
        plt.ylabel('가격 (£)')
        plt.tight_layout()
        
        # 3. MPG와 가격의 관계
        if 'mpg' in self.predictor.processed_df.columns:
            plt.figure(figsize=(15, 6))
            sns.jointplot(data=self.predictor.processed_df, x='mpg', y='price', 
                         kind='hex', height=10)
            plt.suptitle('MPG와 가격의 결합 분포', y=1.02)
            plt.tight_layout()
    
    def create_model_performance_plots(self):
        """모델 성능 관련 시각화"""
        # 1. 예측값과 실제값 비교
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        rf_true = self.model_metrics['RandomForest']['True_Values']
        rf_pred = self.model_metrics['RandomForest']['Predictions']
        plt.scatter(rf_true, rf_pred, alpha=0.5)
        plt.plot([min(rf_true), max(rf_true)], [min(rf_true), max(rf_true)], 'r--')
        plt.title('RandomForest: 실제 가격 vs 예측 가격')
        plt.xlabel('실제 가격 (£)')
        plt.ylabel('예측 가격 (£)')
        
        plt.subplot(1, 2, 2)
        gb_true = self.model_metrics['GradientBoosting']['True_Values']
        gb_pred = self.model_metrics['GradientBoosting']['Predictions']
        plt.scatter(gb_true, gb_pred, alpha=0.5)
        plt.plot([min(gb_true), max(gb_true)], [min(gb_true), max(gb_true)], 'r--')
        plt.title('GradientBoosting: 실제 가격 vs 예측 가격')
        plt.xlabel('실제 가격 (£)')
        plt.ylabel('예측 가격 (£)')
        plt.tight_layout()
        
        # 2. 잔차 분석
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        rf_residuals = rf_true - rf_pred
        sns.histplot(rf_residuals, kde=True)
        plt.title('RandomForest 잔차 분포')
        plt.xlabel('잔차')
        plt.ylabel('빈도')
        
        plt.subplot(1, 2, 2)
        gb_residuals = gb_true - gb_pred
        sns.histplot(gb_residuals, kde=True)
        plt.title('GradientBoosting 잔차 분포')
        plt.xlabel('잔차')
        plt.ylabel('빈도')
        plt.tight_layout()
        
        # 3. 특성 중요도 비교
        rf_importance = pd.DataFrame({
            'feature': self.predictor.features,
            'importance': self.predictor.rf_model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        gb_importance = pd.DataFrame({
            'feature': self.predictor.features,
            'importance': self.predictor.gb_model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        sns.barplot(data=rf_importance, x='importance', y='feature')
        plt.title('RandomForest 특성 중요도')
        plt.xlabel('중요도')
        
        plt.subplot(1, 2, 2)
        sns.barplot(data=gb_importance, x='importance', y='feature')
        plt.title('GradientBoosting 특성 중요도')
        plt.xlabel('중요도')
        plt.tight_layout()
    
    def create_relationship_plots(self):
        """변수 간 관계 시각화"""
        # 1. 상관관계 히트맵
        plt.figure(figsize=(12, 10))
        numeric_cols = self.predictor.processed_df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.predictor.processed_df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('특성 간 상관관계')
        plt.tight_layout()
        
        # 2. 페어플롯 (주요 변수들만)
        main_features = ['price', 'year', 'mileage', 'engineSize']
        if 'mpg' in self.predictor.processed_df.columns:
            main_features.append('mpg')
        sns.pairplot(self.predictor.processed_df[main_features], 
                    diag_kind='kde', plot_kws={'alpha': 0.6})
        plt.suptitle('주요 특성들의 페어플롯', y=1.02)
        plt.tight_layout()
    
    def create_distribution_plots(self):
        """분포 분석 시각화"""
        # 1. 가격 분포 분석
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(data=self.predictor.processed_df, x='price', kde=True)
        plt.title('가격 분포')
        plt.xlabel('가격 (£)')
        plt.ylabel('빈도')
        
        plt.subplot(1, 2, 2)
        sns.histplot(data=self.predictor.processed_df, x=np.log1p(self.predictor.processed_df['price']), kde=True)
        plt.title('로그 변환된 가격 분포')
        plt.xlabel('로그 가격')
        plt.ylabel('빈도')
        plt.tight_layout()
        
        # 2. 브랜드별 가격 분포 변화
        plt.figure(figsize=(15, 8))
        sns.violinplot(data=self.predictor.processed_df, x='brand', y='price')
        plt.title('브랜드별 가격 분포')
        plt.xticks(rotation=45)
        plt.xlabel('브랜드')
        plt.ylabel('가격 (£)')
        plt.tight_layout()
        
        # 3. 연도별-브랜드별 가격 변화
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=self.predictor.processed_df, x='year', y='price', hue='brand')
        plt.title('연도 및 브랜드별 가격 분포')
        plt.xticks(rotation=45)
        plt.xlabel('연도')
        plt.ylabel('가격 (£)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

def main():
    visualizer = EnhancedCarPriceVisualizer()
    visualizer.visualize_all()

if __name__ == "__main__":
    main()