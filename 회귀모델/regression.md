# 회귀(Regression)  
**어떤 자료에 대해서 그 값에 영향을 주는 조건을 고려하여 구한 평균**  
회귀 분석은 데이터 변수들간에 함수관계를 파악하여 통계적 추론을 하는 것  
쉽게 말하면 독립변수(X)에 대한 종속변수(y)값의 평균을 구하는 것  
![image](https://user-images.githubusercontent.com/59672592/126978247-1066bae3-17c5-4db0-a1c8-ca54a49caef5.png)  
- h()가 조건에 따른 평균을 구하는 함수이며 이것을 `회귀 모델`이라고 부름  
    - x,y사이의 관계를 파악한 관계식  
- e는 오차항을 의미  
    - 측정상의 오차나 모든 정보를 파악할 수 없는 점 등 다양한 현실적인 한계로 인해 발생하는 불확실성이 포함됨
    - 이론적으로 평균이 0이고 분산이 정규 분포를 띄는 성질이 있음 
- 잔차(residual)  
    - 데이터의 실축치와 모델의 예측치 사이의 차이. 즉, 회귀식에서 오차항에 대한 관측치
    - 잔차와 독립변수 X 사이에 상관 관계가 없고 자기 자신과도 상관이 없어야 함
- 회귀 분석을 한다는 것은 h()가 무엇인지를 찾는 과정을 의미

---
## 선형 회귀 모델(Linear Regression Model)  
선형 회귀 모델은 **회귀 계수를 선형 결합으로 표현할 수 있는 모델**을 말함  
즉, 독립 변수가 일차식이냐 이차식이냐 로그 함수식이냐가 아니라 우리가 추정할 대상인 파라미터가 어떻게 생겼느냐의 문제  
![image](https://user-images.githubusercontent.com/59672592/126984582-21b7e8f6-7b68-4845-98c8-1c86d6229edf.png)  
- 위의 함수들은 모두 선형 회귀식  
- x^2, x^3 때문에 비선형이라고 생각하기 쉽지만, 회귀 모델의 선형성은 x가 아니라 회귀 계수인 베타0, 베타1, 베타2를 기준으로 생각하는 것이기 때문에 선형 회귀식임  
- 위 식들은 모두 '선형성'에 직접적으로 관련되지 않은 변수인 x와 y를 적절히 변환할 경우 모두 선형 회귀식으로 표현이 가능  
![image](https://user-images.githubusercontent.com/59672592/126984872-f1cfff4a-5f78-493c-b41e-10f816b2ed60.png)  
---
## 비선형 회귀 모델(Non-Linear Regression Model)  
데이터를 어떻게 변형하더라도 파라미터를 선형 결합식으로 표현할 수 없는 모델  

![image](https://user-images.githubusercontent.com/59672592/126985033-cbbce013-350b-4146-8819-d83e5f3ba097.png)  
- 위 식은 아무리 x, y 변수를 변환하더라도 파라미터를 선형식으로 표현할 수 없으므로 비선형식  
---
## 회귀분석 종류  
![image](https://user-images.githubusercontent.com/59672592/126987518-f617f828-6937-43d6-926c-a15cf4a9083d.png)
---
## 고전적 선형 회귀 모델(Classical linear regression model)  
- 단변량 선형 회귀 모델의 가장 기본 형태  
- 독립 변수의 개수에 따라 명칭을 구분
    - 단순 선형 회귀(simple linear regression): 독립 변수가 하나
    - 다중 선형 회귀(multiple linear regression): 독립 변수가 둘 이상
- 데이터에 대한 많은 가정을 가지고 있음
    - 오차항은 평균이 0이고 분산이 일정한 정규 분포를 갖는다
    - 독립변수와 종속변수는 선형 관계이다
    - 오차항은 자기 상관성이 없다
    - 데이터에 아웃라이어(이상치;outlier)가 없다
    - 독립변수와 오차항은 서로 독립이다
    - 독립변수 간에서는 서로 선형적으로 독립이다
따라서 만약 실제 데이터가 이런 가정을 충족하지 않는다면, 고전적 선형 회귀 모델은 실제 데이터를 정확히 반영하지 못하게 되므로 다른 방법을 사용해야 한다. 일반적으로  
- 독립 변수와 종속 변수가 선형 관계가 아닌 경우: Polynomial regression, - - Generalized Additive Model (GAM)
- 오차항의 확률분포가 정규분포가 아닌 경우: Generalized Linear Model (GLM)
- 오차항에 자기 상관성이 있는 경우: Auto-regression
- 데이터에 아웃라이어가 있는 경우: Robust regression, Quantile regression
- 독립변수 간에 상관성이 있는 경우(다중공선성): Ridge regression, Lasso regression, Elastic Net regression, Principal Component Regression (PCR), Partial Least Square (PLS) regression  
## 다중 회귀(Polynomial Regression)  
- 이름 그대로 독립 변수가 다항식으로 구성되는 회귀 모델
- 데이터의 분포의 형태가 직선이 아닌 곡선에 알맞는 모델
![image](https://user-images.githubusercontent.com/59672592/126991761-fa703cf9-40c1-4a33-8815-128a7ba83809.png)  
- 더 낮은 차수의 오차를 얻기 위해 고차 다항식을 사용할 수 있지만, 이는 overfitting을 초래할 수 있음  
![image](https://user-images.githubusercontent.com/59672592/126991895-17b42492-5a74-4675-b332-a31d65d14b93.png)  
## 단계적 회귀(Stepwise Regression)  
- 독립변수를 너무 많이 선택하지 않고 linear regression의 결과가 유의미하도록 multiple linear regression을 수행하기 위해 사용  
- 데이터에 있는 여러개의 독립변수 중에 최소의 독립변수를 선택하여 regression 모델의 예측결과가 데이터에 가장 잘 fitting되도록 하는 방법
- 표준 단계별 회귀는 필요에 따라 각 단계마다 예측 변수를 추가하고 제거함
    - Forward selection: 변수를 하나씩 더해가면서 regression 한 후에 하나씩 제거하면서 regression 결과가 얼마나 영향을 미치는지 보면서 변수를 선택하는 방법  
    - Backward elimination: 모든 변수를 가지고 regression 한 후에 하나씩 제거하면서 regression 결과가 얼마나 영향을 미치는지 보면서 변수를 선택하는 방법
    - Birdirectional procedure: 위 2가지 방법을 병행하여 변수를 선택하거나 제거해가면서 최종 모델을 구하는 방법  
## 로지스틱 회귀(Logistic Regression)  
- True인지 False인지 확률을 찾는데 사용
- 종속변수(Y)가 T/F일때 사용하는 회귀 분석
![image](https://user-images.githubusercontent.com/59672592/126993570-d88502df-8e4c-4ce1-bc33-55e1904d9c79.png)  
- 종속변수 y는 2개의 결과 중 하나를 가짐
- 이항 분포(종속변수(Y))로 작업하기 때문에 이 분포에 가장 적합한 함수가 Logit 함수  
![image](https://user-images.githubusercontent.com/59672592/126993746-39515346-4ccb-4311-9292-d562a59ac81b.png)  
- Classification Problem에 사용
- 종속 변수(Y)와 독립 변수(x) 간의 선형 관계를 요구하지 않음 
- 과도한 피팅과 피팅 부족을 피하려면 모든 중요한 변수를 포함
- 독립 변수간 상관관계가 있으면 안됨 즉 다중 공선성이 없어야함
- 종속 변수의 값이 서수인 경우 서수로 로지스틱 회귀로 호출
- 종속 변수가 다중 클레스이면 다항 로지스틱 회귀를 이용

---
# 다중공선성이 있는 경우  
## Ridge / Lasso / Elastic net regression   
- 다중공선성이 있는 데이터에 대해서 그냥 고전적인 선형 회귀 모델을 만들게 되면 회귀 계수의 영향력이 과다 추정될 수 있음  
    - 다중공선성: 독립 변수간 높은 상관 관계가 있는 것
- 이 문제를 피하기 위해 가장 널리 알려진 방법이 ‘regularization’이라고 부르는 기법 = overfitting 문제 해결 방안  
- Ridge / lasso / elastic net 이 모두 이런 regularization 을 이용한 회귀 모델링 기법  
- 수식으로 표현  
![image](https://user-images.githubusercontent.com/59672592/126994621-82fbb94c-d753-4d6f-a2ba-a5dec936bc46.png)  
- 고전적인 선형 회귀 모델은 회귀 계수를 추정할 때 잔차의 제곱의 합을 계산
- 이 함수를 비용함수라고 부르는데 이 비용 함수가 최소가 되는 회귀 계수를 찾는 것
- Ridge / Lasso / Elastic net regression은 추가적인 수식이 붙는데 이를 페널티 함수라고 함 -> **회귀 계수 값 자체가 너무 커지지 않도록 페널티를 줌으로써 회귀계수값들이 과다 추정되는 것을 막는 것**
- 페널티 함수의 형태에 따라 ridge 와 lasso 가 구분
    - ridge regression 은 회귀 계수의 제곱합을 계산하는 방식이고, 
    - lasso 는 회귀 계수의 절대값을 계산하는 방식
    - elastic net은 이 둘을 결합한 방식
- 이런 페널티 함수를 이용하면 다중공선성이 있더라도 회귀 계수 과다 추정을 막을 수 있으며, 더 나아가 모델이 overfitting 되는 문제도 어느 정도 완화시킬 수 있음
- 보통 독립 변수의 개수가 데이터의 개수에 비해 너무 많은 경우에 이 기법을 사용
- 특히 lasso regression은 영향력이 적은 변수의 회귀 계수값을 0으로 만들기 때문에 일종의 변수 선택 효과까지 있는 장점
### Ridge Regression  
-  λ (λ)를 통해 다중 공선성 문제를 해결
- 계수의 값은 줄이지만 0에 미치지는 않음
- 해당 방식은 Regularization Method이며 L2-Regularization
### Lasso Regression  
- 페널티 인자에서 절대 값을 사용하는 함으로 Ridge Regression과 다름
- 매개 변수 추정 중 일부가 정확히 0이 됨
- 페널티를 더 많이 적용하면 추산치가 0(absolute zero)으로 축소
- 이 회귀의 가정은 정규성을 가정하지 않은 최소 제곱 회귀 분석과 동일하다.
- 계수를 0으로 줄이면 Feature 선택에 도움이 됨
- 해당 방식은 정규화 방법이며 L1-Regularizaion
- 예측 그룹의 상관 관계가 높으면 Lasso는 그중 하나만 선택하고 나머지는 0으로 변경
### Elastic net Regression  
- L1, L2 norm 둘 다 사용하기 때문에 Rodge와 Lasso의 장점을 모두 가지고 있어 변수의 수도 줄고 variance도 줄이고 싶을 때 사용
- ex) 실제로 영향을 주는 변수는 A지만 A와 연관이 있는 변수 B가 있는 경우 통계적으로 B도 영향을 주는 것처럼 보일 수 있음.  
Ridge의 경우 계수를 전체적으로 줄이기 때문에 변수 선택이 안될 것이고, Lasso의 경우 영향을 주지 않는 B가 남고 A의 계수가 0이 되어 없어질 수 있음  
이런 경우에 Elastic net을 사용  
- L1과 L2 norm에 대한 가중치 λ1과 λ2를 조절하여 사용  
---
## 회귀 평가 지표  
- 회귀 평가지표 MAE, MSE, RMSE, MSLE, RMSLE는 값이 작을수록 회귀 성능이 좋은 것 -> 값이 작을수록 예측값과 실제값의 차이가 없다는 뜻이기 때문
-  R² 는 값이 클수록 성능이 좋음
![image](https://user-images.githubusercontent.com/59672592/126996458-48442f41-e323-4dba-85dd-75a3ad6bd093.png)  
- **RMSE (Root Mean Squared Error)**: MSE 값은 오류의 제곱을 구하므로 실제 오류 평균보다 더 커지는 특성이 있어 MSE에 루트를 씌운 RMSE 값을 사용
- **MAE (Mean Absolue Error)**: 실제 값과 예측 값의 차이를 절댓값으로 변환해 평균한 것
- **R² (R Sqaure)**: R² 는 분산 기반으로 예측 성능을 평가, 1에 가까울수록 예측 정확도가 높음  
- **MAE (Mean Absolue Error)**: 실제 값과 예측 값의 차이를 절댓값으로 변환해 평균한 것  
![image](https://user-images.githubusercontent.com/59672592/126997208-d86d4b61-b930-4add-a53b-d5696880ef33.png)  
- **RMSLE (Root Mean Squared Log Error)**: RMSE에 로그를 적용해준 지표
![image](https://user-images.githubusercontent.com/59672592/126997318-bffd9d59-8b9d-4bde-9221-84248bada4a7.png)  

### python에서 사용  
```python
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

origin = np.array([1, 2, 3, 2, 3, 5, 4, 6, 5, 6, 7])
pred = np.array([1, 1, 2, 2, 3, 4, 4, 5, 5, 7, 7])

MAE =  mean_absolute_error(origin, pred)
# MAE = 0.45454545454545453

MSE = mean_squared_error(origin, pred)
# MSE = 0.45454545454545453

RMSE = np.sqrt(MSE)
# RMSE = 0.674199862463242

MSLE = mean_squared_log_error(origin, pred)
# MSLE = 0.029272467607503516

RMSLE = np.sqrt(mean_squared_log_error(origin, pred))
# RMSLE = 0.1710919858073531

R2 = r2_score(origin, pred)
# R2 = 0.868421052631579
```







