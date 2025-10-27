# 배경내용

### Diffuion MRI
용도 : neuroimaging, making the tissue microstructure and structural connections in the in vivo human brain
Diffusion MRI 데이터를 분석하기 위해서 T1w 데이터(이미지?)가 더 필요하다.
하지만 T1w 데이터를 만드는것은 어렵다. 때문에 임의로 합성시키자는 아이디어.

CNN을 이용해서 dMRI로 부터 T1w를 만들어낸다.
세포 이미지 등의 처리에 이미 확실한 성능을 보여주었던 U-net을 generator로 결정하였다.
생성 작업이고, 확실한 이미지를 만들어야 하기 떄문에 GAN 구조를 사용한 것으로 보인다.



### U-net (generator)
$$ R^{64 * 64 * 64 * n} \to R^{64 * 64 * 64} $$

n게 체널의 복셀 이미지(dMRI $R^{64 * 64 * 64}$)를 받아서 T1w이미지로 합성시킨다.    
input: 평균 DWI 볼륨, dMRI의 eigenvector(텐서) 3개, dMRI의 eigenvalue    
맞춰진 텐서로 부터 나온 최적화된 확산 방향을 따르는 6개의 DWI볼륨?     



### CNN (Discriminator)
$$ R^{64 * 64} \to 0,1 $$

복셀 이미지의 단층(2차원) 이미지를 이진 분류한다.



### loss function
$$ L = L_{MAE} + \lambda * L_{adversarial} $$   

람다를 통해서 data fitting loss와 adversarial loss를 적절한 비율로 조정한다.(hyper parameter)   
10^-3 정도를 사용했다.
binary cross entropy 사용



### Network generalization
3가지로 나누어 일반적인 성능을 측정하였다.
1. HCP U-Net: the U-Net trained and validated on the HCP data was directly applied to the MGH-TopupEddy data.  
2. UKB U-Net: the U-Net trained and validated on the UKB data was directly applied to the MGH-TopupEddy data.
3. UKB U-Net (fine-tuned): the U-Net trained and validated on the UKB data was fine-tuned using the MGH-TopupEddy data of three 
additional subjects not used for evaluation and then applied to the MGH-TopupEddy data of the subjects for evaluation.







###### 모르는 용어
> 피질,회백질,백질,섬유 대충 뇌 구조 관련
> 다이스 계수
> dMRI
> DWI
