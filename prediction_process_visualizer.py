import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from regularized_car_price_model import RegularizedCarPricePredictor

class PredictionProcessVisualizer:
    def __init__(self, predictor):
        self.predictor = predictor
        
    def create_prediction_flow(self, input_data):
        """예측 과정을 3D로 시각화"""
        # 1. 데이터 준비
        scaled_data = self.predictor.scaler.transform(input_data[self.predictor.features])
        
        # 2. 각 모델의 예측 과정 시각화
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}],
                  [{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=('데이터 전처리', '모델 예측 과정', 
                          '모델 앙상블', '최종 예측')
        )
        
        # 데이터 전처리 시각화
        self._add_preprocessing_visualization(fig, input_data, scaled_data, 1, 1)
        
        # 모델 예측 과정 시각화
        self._add_prediction_process(fig, scaled_data, 1, 2)
        
        # 앙상블 과정 시각화
        self._add_ensemble_process(fig, scaled_data, 2, 1)
        
        # 최종 예측 시각화
        self._add_final_prediction(fig, scaled_data, 2, 2)
        
        # 레이아웃 설정
        fig.update_layout(
            title='자동차 가격 예측 과정 시각화',
            height=1000,
            showlegend=True
        )
        
        return fig
    
    def _add_preprocessing_visualization(self, fig, raw_data, scaled_data, row, col):
        """데이터 전처리 과정 시각화"""
        # 원본 데이터와 스케일링된 데이터를 3D 공간에 표시
        feature_indices = np.arange(len(self.predictor.features))
        
        # 원본 데이터
        fig.add_trace(
            go.Scatter3d(
                x=feature_indices,
                y=raw_data.iloc[0].values,
                z=np.zeros_like(feature_indices),
                mode='markers',
                name='원본 데이터',
                marker=dict(size=8, color='blue')
            ),
            row=row, col=col
        )
        
        # 스케일링된 데이터
        fig.add_trace(
            go.Scatter3d(
                x=feature_indices,
                y=scaled_data[0],
                z=np.ones_like(feature_indices),
                mode='markers',
                name='스케일링된 데이터',
                marker=dict(size=8, color='red')
            ),
            row=row, col=col
        )
        
        # 전처리 과정을 보여주는 선
        for i in range(len(feature_indices)):
            fig.add_trace(
                go.Scatter3d(
                    x=[feature_indices[i], feature_indices[i]],
                    y=[raw_data.iloc[0].values[i], scaled_data[0][i]],
                    z=[0, 1],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False
                ),
                row=row, col=col
            )
    
    def _add_prediction_process(self, fig, scaled_data, row, col):
        """각 모델의 예측 과정 시각화"""
        # 각 모델의 특성 중요도를 3D 공간에 표시
        models = {
            'RandomForest': self.predictor.rf_model,
            'GradientBoosting': self.predictor.gb_model,
            'Ridge': self.predictor.ridge_model,
            'Lasso': self.predictor.lasso_model
        }
        
        z_positions = np.linspace(0, 1, len(models))
        
        for idx, (name, model) in enumerate(models.items()):
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                importance = np.abs(model.coef_) / np.sum(np.abs(model.coef_))
            
            fig.add_trace(
                go.Scatter3d(
                    x=np.arange(len(self.predictor.features)),
                    y=importance,
                    z=np.full_like(importance, z_positions[idx]),
                    mode='markers+lines',
                    name=f'{name} 특성 중요도',
                    marker=dict(size=8)
                ),
                row=row, col=col
            )
    
    def _add_ensemble_process(self, fig, scaled_data, row, col):
        """앙상블 과정 시각화"""
        # 각 모델의 예측값을 3D 공간에 표시
        predictions = {
            'RandomForest': self.predictor.rf_model.predict(scaled_data)[0],
            'GradientBoosting': self.predictor.gb_model.predict(scaled_data)[0],
            'Ridge': self.predictor.ridge_model.predict(scaled_data)[0],
            'Lasso': self.predictor.lasso_model.predict(scaled_data)[0]
        }
        
        x_positions = np.arange(len(predictions))
        
        fig.add_trace(
            go.Scatter3d(
                x=x_positions,
                y=list(predictions.values()),
                z=np.zeros_like(x_positions),
                mode='markers+text',
                name='모델별 예측값',
                marker=dict(size=10),
                text=list(predictions.keys()),
                textposition='top center'
            ),
            row=row, col=col
        )
    
    def _add_final_prediction(self, fig, scaled_data, row, col):
        """최종 예측 시각화"""
        ridge_pred, lasso_pred, rf_pred, gb_pred, ensemble_pred = \
            self.predictor.predict_price(scaled_data)
        
        predictions = {
            'Ridge': ridge_pred[0],
            'Lasso': lasso_pred[0],
            'RandomForest': rf_pred[0],
            'GradientBoosting': gb_pred[0],
            'Ensemble': ensemble_pred[0]
        }
        
        # 3D 막대 그래프로 표시
        x_positions = np.arange(len(predictions))
        
        fig.add_trace(
            go.Bar3d(
                x=x_positions,
                y=np.zeros_like(x_positions),
                z=list(predictions.values()),
                name='최종 예측값',
                marker=dict(color=['blue', 'green', 'red', 'purple', 'orange'])
            ),
            row=row, col=col
        )

# 사용 예시:
def visualize_prediction_process(predictor, input_data):
    visualizer = PredictionProcessVisualizer(predictor)
    fig = visualizer.create_prediction_flow(input_data)
    fig.show()
    
    # HTML 파일로도 저장
    fig.write_html("prediction_process.html")

# 실행 예시:
if __name__ == "__main__":
    # 예시 데이터
    example_data = pd.DataFrame({
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
    
    predictor = RegularizedCarPricePredictor()
    predictor.train()
    
    visualize_prediction_process(predictor, example_data)