# abstract

# 배경내용
### Diffuion MRI
용도 : neuroimaging, the tissue microstructure and structural connections in the in vivo human brain
가하학적 왜곡 발생할수있음 그런것을 보정해야만 한다.
T1w 데이터를 
하지만 Diffusion MRI 데이터를 분석하기 위해서 아래의 몇 가지 T1w, 데이터가 더 필요하다.
논문에서는 그러한 추가 데이터 


Diffusion MRI 데이터가 있고, T1w 이미지가 존재한다.
Diffusion MRI만으로는 

CNN을 이용해서 Diffusuion MRI데이터를 다루고 

# introduction


### U-net (generator)
$$ R^{64 * 64 * 64 * n} \to R^{64 * 64 * 64} $$

복셀 이미지(dMRI $R^{64 * 64 * 64}$)를 n개 받아서 T1w이미지를 합성한다.
input: 평균 DWI 볼륨, dMRI의 eigenvector(텐서) 3개, dMRI의 eigenvalue
맞춰진 텐서로 부터 나온 최적화된 확산 방향을 따르는 6개의 DWI볼륨



### CNN (Discriminator)
$$ R^{64 * 64} \to 0,1 $$


### loss function
L = L_MAE + \lambda * L_adversarial
binary cross entropy 사용

# 읽은 곳 까지 정리'


# 모르는 용어
피질,회백질,백질,섬유 대충 뇌 구조 관련
다이스 계수
dMRI
DWI
