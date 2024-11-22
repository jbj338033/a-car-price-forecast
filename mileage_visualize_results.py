import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from car_price_model import CarPricePredictor
import matplotlib.font_manager as fm
import platform
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class MileageAnalysisVisualizer:
    def __init__(self):
        # 한글 폰트 설정
        if platform.system() == 'Darwin':  # macOS
            plt.rc('font', family='AppleGothic')
        plt.rc('axes', unicode_minus=False)
        
        self.predictor = CarPricePredictor()
        self.set_style()
        
        # 브랜드 매핑
        self.brand_mapping = {
            'audi': 'Audi', 'bmw': 'BMW', 'cclass': 'Mercedes C-Class',
            'focus': 'Ford Focus', 'ford': 'Ford', 'hyundi': 'Hyundai',
            'merc': 'Mercedes', 'skoda': 'Skoda', 'toyota': 'Toyota',
            'vauxhall': 'Vauxhall', 'vw': 'Volkswagen'
        }
    
    def set_style(self):
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
    
    def prepare_data(self):
        """데이터 준비 및 전처리"""
        print("데이터 준비 중...")
        model_metrics = self.predictor.train()
        
        # 데이터 복사 및 브랜드 이름 매핑
        df = self.predictor.processed_df.copy()
        df['brand_name'] = df['brand'].map({idx: name for idx, name in 
                                          enumerate(self.brand_mapping.values())})
        
        # 주행거리 구간 생성
        df['mileage_group'] = pd.qcut(df['mileage'], q=5, 
                                    labels=['매우 낮음', '낮음', '보통', '높음', '매우 높음'])
        
        return df, model_metrics
    
    def analyze_mileage_relationships(self, df):
        """주행거리와 다른 변수들의 관계 분석"""
        print("\n주행거리 관계 분석 중...")
        
        # 1. 주행거리와 가격의 관계 (전체)
        plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=df, x='mileage', y='price', alpha=0.5)
        plt.title('주행거리와 가격의 관계')
        plt.xlabel('주행거리')
        plt.ylabel('가격 (£)')
        
        # 다항 회귀 피팅
        X = df['mileage'].values.reshape(-1, 1)
        y = df['price'].values
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_plot_poly = poly.transform(X_plot)
        y_plot = model.predict(X_plot_poly)
        plt.plot(X_plot, y_plot, 'r-', label='추세선')
        plt.legend()
        
        # 주행거리 구간별 평균 가격
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x='mileage_group', y='price')
        plt.title('주행거리 구간별 가격 분포')
        plt.xlabel('주행거리 구간')
        plt.ylabel('가격 (£)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 2. 브랜드별 주행거리와 가격의 관계
        plt.figure(figsize=(15, 10))
        sns.lmplot(data=df, x='mileage', y='price', hue='brand_name', 
                  height=8, aspect=1.5, scatter_kws={'alpha':0.5})
        plt.title('브랜드별 주행거리와 가격의 관계')
        plt.tight_layout()
        
        # 3. 연식별 주행거리 분포
        plt.figure(figsize=(15, 6))
        sns.boxplot(data=df, x='year', y='mileage')
        plt.title('연식별 주행거리 분포')
        plt.xlabel('연도')
        plt.ylabel('주행거리')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 4. 주행거리 구간별 통계
        mileage_stats = df.groupby('mileage_group').agg({
            'price': ['mean', 'median', 'count'],
            'year': 'mean',
            'engineSize': 'mean'
        })
        print("\n주행거리 구간별 통계:")
        print(mileage_stats)
        
        # 5. 주행거리와 연비(mpg)의 관계
        if 'mpg' in df.columns:
            plt.figure(figsize=(15, 6))
            sns.scatterplot(data=df, x='mileage', y='mpg', hue='brand_name', alpha=0.5)
            plt.title('주행거리와 연비의 관계')
            plt.xlabel('주행거리')
            plt.ylabel('연비 (MPG)')
            plt.legend(bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
        
        # 6. 주행거리 분포 분석
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(data=df, x='mileage', bins=50, kde=True)
        plt.title('주행거리 분포')
        plt.xlabel('주행거리')
        plt.ylabel('빈도')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x='brand_name', y='mileage')
        plt.title('브랜드별 주행거리 분포')
        plt.xticks(rotation=45)
        plt.xlabel('브랜드')
        plt.ylabel('주행거리')
        plt.tight_layout()
        
        # 7. 연료 타입별 주행거리 분석
        plt.figure(figsize=(15, 6))
        sns.boxplot(data=df, x='fuelType', y='mileage')
        plt.title('연료 타입별 주행거리 분포')
        plt.xlabel('연료 타입')
        plt.ylabel('주행거리')
        plt.tight_layout()
        
        # 8. 주행거리와 가격의 상관관계 분석
        correlation = df['mileage'].corr(df['price'])
        print(f"\n주행거리와 가격의 상관계수: {correlation:.4f}")
        
        # 9. 주행거리 구간별 브랜드 분포
        plt.figure(figsize=(15, 8))
        brand_mileage = pd.crosstab(df['brand_name'], df['mileage_group'])
        brand_mileage_pct = brand_mileage.div(brand_mileage.sum(axis=1), axis=0)
        brand_mileage_pct.plot(kind='bar', stacked=True)
        plt.title('브랜드별 주행거리 구간 분포')
        plt.xlabel('브랜드')
        plt.ylabel('비율')
        plt.legend(title='주행거리 구간', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        
        # 10. 주행거리 이상치 분석
        Q1 = df['mileage'].quantile(0.25)
        Q3 = df['mileage'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df['mileage'] < (Q1 - 1.5 * IQR)) | (df['mileage'] > (Q3 + 1.5 * IQR))]
        
        print(f"\n주행거리 이상치 분석:")
        print(f"이상치 수: {len(outliers)}")
        print(f"전체 데이터 대비 이상치 비율: {(len(outliers)/len(df))*100:.2f}%")
        
        plt.figure(figsize=(15, 6))
        sns.scatterplot(data=df, x='mileage', y='price', alpha=0.5)
        sns.scatterplot(data=outliers, x='mileage', y='price', color='red', label='이상치')
        plt.title('주행거리 이상치와 가격의 관계')
        plt.xlabel('주행거리')
        plt.ylabel('가격 (£)')
        plt.legend()
        plt.tight_layout()
        
        # 11. 연도별 평균 주행거리 트렌드
        plt.figure(figsize=(15, 6))
        yearly_mileage = df.groupby('year')['mileage'].agg(['mean', 'std']).reset_index()
        plt.errorbar(yearly_mileage['year'], yearly_mileage['mean'], 
                    yerr=yearly_mileage['std'], capsize=5)
        plt.title('연도별 평균 주행거리 추이')
        plt.xlabel('연도')
        plt.ylabel('평균 주행거리')
        plt.tight_layout()
        
        return mileage_stats

def main():
    analyzer = MileageAnalysisVisualizer()
    df, _ = analyzer.prepare_data()
    mileage_stats = analyzer.analyze_mileage_relationships(df)
    plt.show()

if __name__ == "__main__":
    main()