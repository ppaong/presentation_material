# 진행척도
3.ML기초실습 : 진행중



# ML기초실습

## numpy.linalg as la
numpy는 행렬을 다루기 편하도록 여러 선형대수학의 함수들을 지원한다.

### 단위행렬
$I_n$ : la.eye(n)


### Singular value decomposition (SVD)
>  

$A=U \Sigma V^T$ : U, S, V = la.svd(A)
기본적으로 full 형태의 행렬을 반한하며, la.svd(A,full_matrix=False)으 reduced SVD를 실행.


### QR decomposition
> A의 column vector를 Gram_Schmidt process를 통해 othogonal vector를 얻어내어 다시 A를 나타내는 방식(Q:othogonal matrix)

$A=QR$ : la.qr(A)

la.qr(A,mode='reduced') 하면 Q,R을 reduced for으로 반환


### Cholesky factorication
> LU분해인데 positive definite에서 특이한 경우

$A=LL^T$ : L = la.cholesky()


### Eigen value decomposition
> 

$A=V \Lambda V^-1$ : L,V=la.eig(1)


### Inverse

$A^-1$ : la.inv(A)
$A^+$ : la.pinv(A) *(n*m 행렬 A의 pseudo inverse)









## pandas
데이터를 다루는 메소드를 지원하는 패키기
np랑 다른점은
* file io를 중점으로 다룬다.
* missing data같은 데이터 처리에 특화
* time series data

간단히 데이터를 구조적으로 다룰때 pandas쓰고, 데이터 연산처리할떄 np 쓴다고 이해.


## scikit learn

## pytorch
