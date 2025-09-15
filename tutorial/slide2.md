# 진행척도
3.ML기초실습 : 진행중



# ML기초실습

## numpy.linalg as la
numpy는 행렬을 다루기 편하도록 여러 선형대수학의 함수들을 지원한다.


### 단위행렬
$I_n$ : la.eye(n)


### Singular value decomposition (SVD)
> V의 직교좌표계를 A를 통해 선형변환 했을때, 직교성질이 유지되는 직교좌표계 U를 찾아낼 수 있다는 해석이 있다. othogonal vector로 이루어진 행렬의 성질을 이용해 그러한 관계를 아래와 같이 한번에 나타낼수있다.

$A=U \Sigma V^T$ : U, S, V = la.svd(A)   
기본적으로 full 형태의 행렬을 반한하며, la.svd(A,full_matrix=False)으 reduced SVD를 실행. (U의 column space의 차원이 낮은경우 가능하다.)


### QR decomposition
> A의 column vector를 Gram_Schmidt process를 통해 othogonal vector를 얻어내어 다시 A를 나타내는 방식(Q:othogonal matrix)

$A=QR$ : la.qr(A)

la.qr(A,mode='reduced') 하면 Q,R을 reduced for으로 반환


### Cholesky factorication
> LU분해인데 positive definite에서 특이한 경우 빠른 분해가 가능하다.

$A=LL^T$ : L = la.cholesky(A)


### Eigen value decomposition
> 

$A=V \Lambda V^-1$ : L,V = la.eig(A)   
$A=V \Lambda V^T$ : L,V = la.eigh(A) *(대칭인 경우 더 빠른 분해)  
eigen value 만 구하고싶은 경우 V = la.eigvals(A) 로 구할 수 있다.


### Inverse

$A^-1$ : la.inv(A)   
$A^+$ : la.pinv(A) *(n*m 행렬 A의 pseudo inverse)






## scipy
유용한 수학 과학 관련 함수를 지원하는 패키지

### interpolate
> 보간법은 몇개의 주어진 점을 잘 연걸하는 선을 그리는 벙법이다.

import scipy.interpolate as interp
f = interp.interp1d(x,y,kind="zero") 를 통해 함수 선언을 할 수 있다. 
attribute 중 kind 는 "linear", "cubic" 등 다양하게 지원한다.


### optimize

from scipy import optimize
results = optimize.minimize(f,x0=(0,0))


## pandas
데이터를 다루는 메소드를 지원하는 패키기
np랑 다른점은
* file io를 중점으로 다룬다.
* missing data같은 데이터 처리에 특화
* time series data

간단히 데이터를 구조적으로 다룰때 pandas쓰고, 데이터 연산처리할떄 np 쓴다고 이해.


## scikit learn

## pytorch
