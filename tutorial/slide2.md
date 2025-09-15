# 진행척도
2. 파이썬 기초실습 : 완료
3. ML기초실습 : 진행중
lecture 4,5 완료

개념 복습및 이해 위주로 정리했습니다.




## numpy.linalg as la
numpy는 행렬을 다루기 편하도록 여러 선형대수학의 함수들을 지원한다.


### 단위행렬
$I_n$ : la.eye(n)


### Inverse

$A^{-1}$ : la.inv(A)   
$A^+$ : la.pinv(A)  (m*n 행렬 A의 pseudo inverse)


### Cholesky factorication
> LU분해인데 positive definite에서 특이한 경우 빠른 분해가 가능하다.

$A=LL^T$ : L = la.cholesky(A)


### QR decomposition
> A의 column vector를 Gram_Schmidt process를 통해 othogonal bases를 얻어내어 다시 A를 나타내는 방식(Q:othogonal matrix)

$A=QR$ : la.qr(A)

la.qr(A,mode='reduced') 하면 Q,R을 reduced form으로 반환


### Singular value decomposition (SVD)
> $AV = U\Sigma$ 를 통해 생각해보면, 직교좌표계V를 A를 통해 선형변환 했을때, 직교성질이 유지되는 좌표계 U와 적절한 scaler Sigma를 찾아낼 수 있다고 해석할 수 있다. othogonal vector로 이루어진 행렬의 성질을 이용해 그러한 관계를 아래와 같이 한번에 나타낼수있다.

$A=U \Sigma V^T$ : U, S, V = la.svd(A)   
기본적으로 full 형태의 행렬을 반한하며, la.svd(A,full_matrix=False)으 reduced SVD를 실행. (U의 column space의 차원이 낮은경우 가능하다.)


### Eigen value decomposition (EVD)
> 정방행렬 A에 대한 Eigen vector 의 성질 $Av_n = \lambda v_n$ 을 이용해  $AV=V\Lambda$ 형태를 A에 대해여 다음의 관계식으로 니티낼 수 있다.

$A=V \Lambda V^{-1}$ : L,V = la.eig(A)   
$A=V \Lambda V^T$ : L,V = la.eigh(A) *(대칭인 경우 더 빠른 분해)  
eigen value 만 구하고싶은 경우 V = la.eigvals(A) 로 구할 수 있다.


### SVD와 EVD에서 중요한 성질
구조적으로 중앙의 diagonal matrix를 두고 양쪽에 othogonal vector로 이루어진 martix가 곱해진 형태이다. 이는 중요한 성질을 내포한다.
이는 행렬을 대각화 했다고 표현한다.

나중에 PAC할때 이를 중심적으로 사용하는데, 데이터의 standard 좌표계보다 데이터를 더 잘 나타내는 직교좌표계를 찾는 기법이다.
아에 공분산 행렬을 대각화 시켜서 주요 주성분을 나타내는 othogonal vector를 뽑아낼 수 있다. (정방행렬에 대칭이므로 EVD를 사용)





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
위의 코드는 정의된 함수f를 초기값 x0부터 시작해서 최소값을 찾아간다.   
수치해석 기법으로 BFGS를 사용해서 정확한 값을 구한다.   
BFGS는 수렴속도는 빠르지만, 계산랑이 많은 quasi-newton method의 효율적인 버전 이다. (hessian 역행렬 계산량을 효과적으로 줄였다.)    
local 한 최소지점을 찾아간다. (전역적으로 최소는 아니다.)(실험 자료를 첨부하지 못했습니다.)    


### curve fitting
> 주어진 함수f로 점을 가장 잘 표현하는

def f(x,a,b):
  ~~a,b를 파라메터로 가지는 f(x|a,b) 로 정의
((a,b),_)=optimize.curve.fit(f,x,y,p0=(0,0))
x,y는 점 집합. p0는 시작점.

점집합(데이터셋)을 가장 잘 표현하는 f의 파라메터값을 반환해준다.


### root finding
> 정의된 연속 함수f에 대해 가장 가까운 방향(떨어지는 방향)의 zero를 탐색

optimize.root(f,x0=(0,0))    
초기값 x0부터 시작해서 근방의 zero를 얻어낼수있다.   


### linear programming
> convex한 최적화 문제에서 사용할 수 있는 선형계획법을 제공

$$ \begin{equation}
\begin{split}
\text{minimize} \;\; & c^{T}x  \\
\text{subject to} \;\; & A_{ub}x \leq b_{ub} \\
& A_{eq}x = b_{eq}
\end{split}
\end{equation} $$

과 같이 표현되는 최적화 문제를    
linprog(c, A_ub=, b_ub=, A_eq=, b_eq=, bounds=(0, None))   
에 대입하여 풀어낼 수 있다.




### integration
> 적분및 ode를 풀수있는 연산들을 제공

$$ \frac{dy}{dt} = f(y,t) $$

t = np.linspace(0,10,100)   
Y = integ.odeint(f,[1,1],t)    (주어진 초기값 y=[1,1] 에 대해, 구간 t in [0,10] 에서 ode의 해 y를 구해서 반환합니다.)    
각종 물리문제에 접했을때 유용하게 쓸수있을 것 같습니다.



## 그 외 유용한 패키지 
### networkx
> 그래프(node,edge가 포함된)을 쉽게 그릴수있는 패키지

### sympy
> 써놓은 수식 코드를 latex로 출력해주는 패키지






