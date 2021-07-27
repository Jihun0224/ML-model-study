# Data Scaling  
- 데이터 스케일링은 **데이터의 값의 범위를 조정하는 것**
- ex) x1이 0 ~ 1 사이의 값을 갖고  
x2는 1000000 ~ 100000000 사이의 값을 갖고  
y가 1000000 ~ 1000000000 사이의 값을 갖는다면  
x1 특성은 y를 예측하는데 큰 영향을 주지 않는 것으로 판단할 수 있음  
    - 이 외에도 자료의 오버플로우(overflow)나 언더플로우(underflow)를 방지 하고, 최적화 과정에서 안정성 및 수렴 속도를 향상시킴  
---
## :heavy_exclamation_mark:주의 해야할 점  

- 모든 스케일링은 테스트 데이터가 포함된 전체 데이터셋이 아닌 오로지 훈련 데이터에 대해서만 fit 되어야함
- 이후 훈련 데이터와 테스트 데이터 각각을 스케일링
    - **fit_transform()** 는 train dataset에서만 사용
    - train data에 있는 mean과 variance를 학습
    - **transform()** 는 train data로부터 학습된 mean값과 variance값을 test data에 적용하기 위해 사용
        - 만약에 fit_transform을 test data에도 적용하게 된다면 test data로부터 새로운 mean값과 variance값을 얻게 되는 것
        - 이 데이터마저 학습하게 된다면 우리의 모델이 처음 보는 데이터에 대해서 얼마나 성능이 좋은지 알 수 없게 되는 것
        - test data는 모델이 학습된 후에 평가할 때만 사용되어야 하는데 **fit_transform을 test data에도 하게 된다면 모델의 성능을 평가할 수 없게 되는 것**  
- 모델이 학습을 하는 과정에서 어떠한 방식으로든 테스트 데이터를 사용하게 되어서는 안됨
- 일반적으로 타겟(y) 데이터에 대한 스케일링은 진행하지 않음
- 많은 스케일러들이 이상치의 영향을 많이 받음
    - 영향을 최소화한 RobustScaler가 있지만 이상치를 제거해 주는 것이 훨씬 좋음
- 모든 특성의 범위를 유사하게 만드는 것은 중요하지만, 그렇다고 반드시 같은 분포로 만들 필요는 없음
- sklearn 모듈을 사용하면 데이터프레임 전체를 한번에 변환가능하나, 특성에 따라 각기 다른 스케일링을 적용하는게 유리할 경우가 있음
---
## 데이터 스케일링 종류  
### 1.StandardScaler  
- 각 특성의 평균을 0, 분산을 1로 스케일링
- 즉, 데이터를 정규분포로 만듦
- 최솟값과 최댓값의 크기를 제한하지 않기 때문에, 어떤 알고리즘에서는 문제가 있음
- 회귀보다 분류에 유용  
```python
from sklearn.preprocessing import StandardScaler

# 변형 객체 생성
std_scaler = StandardScaler()

# 훈련데이터의 모수 분포 저장
std_scaler.fit(X_train)

# 훈련 데이터 스케일링
X_train_scaled = std_scaler.transform(X_train)

# 훈련 데이터 모수 분포 저장 및 스케일링
# X_train_scaled = std_scaler.fit_transform(X_train)

# 테스트 데이터의 스케일링
X_test_scaled = std_scaler.transform(X_test)

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있음
# X_origin = std_scaler.inverse_transform(X_train_scaled)
```
---
### 2.RobustScaler  
- 평균과 분산 대신에 중간 값과 사분위 값을 사용
    - 중간 값은 정렬시 중간에 있는 값을 의미하고,
    - 사분위값은 1/4, 3/4에 위치한 값을 의미
- 전체 데이터와 아주 동떨어진 데이터(이상치)에 영향을 받지 않음  
```python
from sklearn.preprocessing import RobustScaler

# 변형 객체 생성
robust_scaler = RobustScaler()

# 훈련데이터의 모수 분포 저장
robust_scaler.fit(X_train)

# 훈련 데이터 스케일링
X_train_scaled = robust_scaler.transform(X_train)

# 훈련 데이터 모수 분포 저장 및 스케일링
# X_train_scaled = robust_scaler.fit_transform(X_train)

# 테스트 데이터의 스케일링
X_test_scaled = robust_scaler.transform(X_test)

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있다.
# X_origin = robust_scaler.inverse_transform(X_train_scaled)
```
---
### 3. MinMaxScaler  
- 각 특성이 0과 1 사이에 위치하도록 스케일링
- 분류보다 회귀에 유용  
```python
from sklearn.preprocessing import MinMaxScaler

# 변형 객체 생성
minmax_scaler = MinMaxScaler()

# 훈련데이터의 모수 분포 저장
minmax_scaler.fit(X_train)

# 훈련 데이터 스케일링
X_train_scaled = minmax_scaler.transform(X_train)

# 훈련 데이터 모수 분포 저장 및 스케일링
# X_train_scaled = minmax_scaler.fit_transform(X_train)

# 테스트 데이터의 스케일링
X_test_scaled = minmax_scaler.transform(X_test)

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있다.
# X_origin = minmax_scaler.inverse_transform(X_train_scaled)
```
---
### 4. MaxAbsScaler  
- 각 특성의 절대값이 0과 1 사이가 되도록 스케일링
- 즉, 모든 값은 -1과 1 사이로 표현되며, 데이터가 양수일 경우 MinMaxScaler와 같음  
```python 
from sklearn.preprocessing import MaxAbsScaler

# 변형 객체 생성
maxabs_scaler = MaxAbsScaler()

# 훈련데이터의 모수 분포 저장
maxabs_scaler.fit(X_train)

# 훈련 데이터 스케일링
X_train_scaled = maxabs_scaler.transform(X_train)

# 훈련 데이터 모수 분포 저장 및 스케일링
# X_train_scaled = maxabs_scaler.fit_transform(X_train)

# 테스트 데이터의 스케일링
X_test_scaled = maxabs_scaler.transform(X_test)

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있다.
# X_origin = maxabs_scaler.inverse_transform(X_train_scaled)
```
---
### 5. Normalizer  
- 앞의 4가지 스케일러는 각 특성(열)의 통계치를 이용하여 진행.
- 그러나 Normalizer의 경우 각 샘플(행)마다 적용되는 방식
- 이는 한 행의 모든 특성들 사이의 유클리드 거리(L2 norm)가 1이 되도록 스케일링
- 일반적인 데이터 전처리의 상황에서 사용되는 것이 아니라 모델(특히나 딥러닝) 내 학습 벡터에 적용하며, 특히나 피쳐들이 다른 단위(키, 나이, 소득 등)라면 더더욱 사용하지 않음  
```python
from sklearn.preprocessing import Normalizer

# 변형 객체 생성
normal_scaler = Normalizer()

# 훈련데이터의 모수 분포 저장
normal_scaler.fit(X_train)

# 훈련 데이터 스케일링
X_train_scaled = normal_scaler.transform(X_train)

# 훈련 데이터 모수 분포 저장 및 스케일링
# X_train_scaled = normal_scaler.fit_transform(X_train)

# 테스트 데이터의 스케일링
X_test_scaled = normal_scaler.transform(X_test)

# 스케일링 된 결과 값으로 본래 값을 구할 수도 있다.
# X_origin = normal_scaler.inverse_transform(X_train_scaled)
```
### :heavy_exclamation_mark: 정규화(Normalization)와 표준화(Standardization)  
- **정규화(Normalization)** 는 값들을 특정 범위, 주로 [0,1]로 스케일링하는 것을,
- **표준화(Standardization)** 는 평균을 0, 표준편차를 1이 되도록 값을 스케일링하는 것을 의미
