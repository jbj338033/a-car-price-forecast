import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from car_price_model import CarPricePredictor
import matplotlib.font_manager as fm
import platform
from scipy import stats

class BrandAnalysisVisualizer:
    def __init__(self):
        # 한글 폰트 설정
        if platform.system() == 'Darwin':  # macOS
            plt.rc('font', family='AppleGothic')
        plt.rc('axes', unicode_minus=False)
        
        self.predictor = CarPricePredictor()
        self.set_style()
        
        # 브랜드 매핑 딕셔너리 생성
        self.brand_mapping = {
            'audi': 'Audi',
            'bmw': 'BMW',
            'cclass': 'Mercedes C-Class',
            'focus': 'Ford Focus',
            'ford': 'Ford',
            'hyundi': 'Hyundai',
            'merc': 'Mercedes',
            'skoda': 'Skoda',
            'toyota': 'Toyota',
            'vauxhall': 'Vauxhall',
            'vw': 'Volkswagen'
        }
        
    def set_style(self):
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
    
    def prepare_data(self):
        """데이터 준비 및 브랜드 이름 매핑"""
        print("데이터 준비 중...")
        model_metrics = self.predictor.train()
        
        # 브랜드 이름 매핑
        df = self.predictor.processed_df.copy()
        df['brand_name'] = df['brand'].map({idx: name for idx, name in 
                                          enumerate(self.brand_mapping.values())})
        return df, model_metrics
    
    def analyze_brand_impact(self, df):
        """브랜드가 가격에 미치는 영향 분석"""
        print("\n브랜드 영향 분석 중...")
        
        # 1. 브랜드별 평균 가격 분석
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=df, x='brand_name', y='price', order=df.groupby('brand_name')['price'].median().sort_values(ascending=False).index)
        plt.title('브랜드별 가격 분포')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('브랜드')
        plt.ylabel('가격 (£)')
        plt.tight_layout()
        
        # 2. ANOVA 테스트
        f_statistic, p_value = stats.f_oneway(*[group['price'].values for name, group in df.groupby('brand_name')])
        print(f"\nANOVA 테스트 결과:")
        print(f"F-통계량: {f_statistic:.2f}")
        print(f"p-값: {p_value:.10f}")
        if p_value < 0.05:
            print("브랜드는 가격에 통계적으로 유의미한 영향을 미칩니다.")
        
        # 3. 브랜드별 상세 통계
        brand_stats = df.groupby('brand_name')['price'].agg([
            ('평균 가격', 'mean'),
            ('중앙값', 'median'),
            ('표준편차', 'std'),
            ('최소값', 'min'),
            ('최대값', 'max'),
            ('데이터 수', 'count')
        ]).round(2)
        
        print("\n브랜드별 가격 통계:")
        print(brand_stats)
        
        # 4. 브랜드별 연도에 따른 가격 트렌드
        plt.figure(figsize=(15, 8))
        for brand in df['brand_name'].unique():
            brand_data = df[df['brand_name'] == brand]
            yearly_avg = brand_data.groupby('year')['price'].mean()
            plt.plot(yearly_avg.index, yearly_avg.values, marker='o', label=brand)
        
        plt.title('브랜드별 연도에 따른 평균 가격 변화')
        plt.xlabel('연도')
        plt.ylabel('평균 가격 (£)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # 5. 브랜드별 엔진 크기와 가격의 관계
        plt.figure(figsize=(15, 8))
        for brand in df['brand_name'].unique():
            brand_data = df[df['brand_name'] == brand]
            plt.scatter(brand_data['engineSize'], brand_data['price'], 
                       alpha=0.5, label=brand)
        
        plt.title('브랜드별 엔진 크기와 가격의 관계')
        plt.xlabel('엔진 크기')
        plt.ylabel('가격 (£)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # 6. 브랜드별 주행거리와 가격의 관계
        plt.figure(figsize=(15, 8))
        sns.lmplot(data=df, x='mileage', y='price', hue='brand_name', 
                  height=8, aspect=1.5, scatter_kws={'alpha':0.5})
        plt.title('브랜드별 주행거리와 가격의 관계')
        plt.tight_layout()
        
        # 7. 브랜드별 가격 분포 (바이올린 플롯)
        plt.figure(figsize=(15, 8))
        sns.violinplot(data=df, x='brand_name', y='price', 
                      order=df.groupby('brand_name')['price'].median().sort_values(ascending=False).index)
        plt.title('브랜드별 가격 분포 (바이올린 플롯)')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('브랜드')
        plt.ylabel('가격 (£)')
        plt.tight_layout()
        
        # 8. 브랜드별 연료 타입 분포
        plt.figure(figsize=(15, 8))
        fuel_type_counts = pd.crosstab(df['brand_name'], df['fuelType'])
        fuel_type_props = fuel_type_counts.div(fuel_type_counts.sum(axis=1), axis=0)
        fuel_type_props.plot(kind='bar', stacked=True)
        plt.title('브랜드별 연료 타입 분포')
        plt.xlabel('브랜드')
        plt.ylabel('비율')
        plt.legend(title='연료 타입', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        
        # 9. 고가 자동차 비율 분석
        price_threshold = df['price'].quantile(0.75)  # 상위 25% 기준
        high_end_ratio = df.groupby('brand_name').apply(
            lambda x: (x['price'] > price_threshold).mean()
        ).sort_values(ascending=False)
        
        plt.figure(figsize=(15, 6))
        high_end_ratio.plot(kind='bar')
        plt.title(f'브랜드별 고가 자동차 비율 (상위 25% 기준: £{price_threshold:,.0f})')
        plt.xlabel('브랜드')
        plt.ylabel('고가 자동차 비율')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 10. 브랜드별 가격대 분포
        price_ranges = pd.qcut(df['price'], q=5, labels=['매우 낮음', '낮음', '중간', '높음', '매우 높음'])
        df['price_range'] = price_ranges
        
        plt.figure(figsize=(15, 8))
        price_range_counts = pd.crosstab(df['brand_name'], df['price_range'])
        price_range_props = price_range_counts.div(price_range_counts.sum(axis=1), axis=0)
        price_range_props.plot(kind='bar', stacked=True)
        plt.title('브랜드별 가격대 분포')
        plt.xlabel('브랜드')
        plt.ylabel('비율')
        plt.legend(title='가격대', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        
        return brand_stats

def main():
    analyzer = BrandAnalysisVisualizer()
    df, _ = analyzer.prepare_data()
    brand_stats = analyzer.analyze_brand_impact(df)
    plt.show()

if __name__ == "__main__":
    main()