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



이번주 듀토리얼 진행소식을 알려드리겠습니다.

이번주는 듀토리얼 7장 논문읽기 파트를 진행했고,
읽은 논문명은 ~~입니다.

내용 이해보다는 모델 돌려보는 부분을 집중해서 공부했습니다.

먼저 제가 이해한 논문 내용은 다음과 같습니다.

Diffuction MRI 라는게 있는데(굉장히 멋지게 생겼습니다.) 이게 쫌 측정하기도 좋고, 만들기도 쉽고 해서 굉장히 좋은 데이터인데. 이것만 보고는 진단하기 어렵다. 그래서 추가적으로 t1w 뇌 스캔 이미지가 팔요한데 이게 정확히 측정하는게(만들어내는게) 어려운 일이다. 환자의 작은 움직임에도 오차가 발생하고 사진 찍는 시간도 굉장히 오래걸린다고 한다. 그래서 논문 저자는 Diffusion MRI 데이터를 가지고 t1w이미지를 합성(생성)해보자고 제안합니다. 여기서 제안하는 모델 이름이 DeepAnat 이고 구현은 tensorflow 구버전으로 구현되어있습니다.

DeepAnat모델의 구조는 일단 입력으로 복셀 이미지(3차원 이미지)(DiffusionMRI 데이터 형식)을 받아서, 같은 형식으로 출력하는 output을 가졌습니다.3D Unet을 사용해서 위와 같은 모델을 설계했고, 학습방식은 GAN방식으로 구현했습니다.
여기서 특이사항은 generator로 사용된 Unet 모델을 3D 이고 discriminator로 사용된 모델은 2D이미지를 받는다는 점입니다.
저자는 ~~~연구를 근거로 들며, GAN학습방식을 사용할때 discriminator는 2D가 좋다고 합니다.

논문 내용은 여기까지이고 github에 저자가 구현해둔 DeepAnat파일을 실행시켜본 내용은 다음과 같습니다.

모델을 돌린 환경은
원도우 11
TITAN X (12GB)
CUDA 10.1
cudnn 7.5.0

tensorflow 구버전과 호환을 위해 낮은 버전을 깔고 환경설정을 했습니다.

아나콘다를 사용해서 python 3.7버전에서
tensorflow-gpu==2.1.0
keras==2.3.1
구버전과 그외 추가적인 패키지를 설치하여 환경설정을 완료했습니다.

구버전의 tensorflow와 numpy 최신버전은 충돌이 일어나는것 같습니다. 때문에 numpy도 구버전으로 다시 깔았습니다.

conda로 까는것과 pip으로 까는것은 약간의 차이가 있는것 같습니다.

import 부분에서 계속 오류가 나서 콘다 환경을 삭제하고 패키지 다시깔기를 계속하며 얻어낸 결과인데, vscode 에서 주피터를 실행하여 커널을 찾을때는 vscode를 종료했다가 다시 시작하고 커넬을 찾는게 좋습니다. 잘못하면 과거에 삭제된 환경을 vscode에서 연결시킬수 있습니다.

여기까지 학습 이전의 문제점들 이였고, 학습하며 생긴 문제점은.

저장부분에서 계속 h5py패키지가 깔려있음에도 import하지 못하는 문제가 있어서 체크포인트마다 학습이 멈추는 문제가 있었습니다. tensorflow 구버전에서는 h5py를 삭제하고 다시 구버전으로 다시 깔아줘야 하는것 같습니다.

추가적으로 배치 사이즈가 작고, 학습 속도가 느려서 멀티 GPU로 돌려보려고 여러 차례 시도했으나, 구버전 cudnn파일의 문제인지 자꾸 No OpKernel was registered to support Op 'NcclAllReduce' 에러가 뜨며 어떤 연산이 누락되서 실행할 수 없었습니다. (8.0.5 ~ 7.6.0 , 7.5.0 버전의 cudnn을 가져와서 시도해보았으나 전부 같은 에러가 떴습니다. 도저히 고칠수 없어서 싱글 GPU를 사용하고 일단 넘어갔습니다.)

확실히 배운것은 구버전의 패키지를 사용할때는 나머지도 다 구버전으로 다시 깔아야 한다는 점을 배웠습니다.

대부분 문제를 해결하고 학습을 돌려봤습니다.
batchsize = 1
epoch = 30
learning rate =

으로 설정되었고 시간은 총 ?? 걸렸습니다.
loss는 아래와 같이 나왔습니다.

다음날 loss그래프를 띄우고 모델에 데이터를 넣어서 샘플 이미지를 뽑아보고 싶었으나, 무슨 문제인지 모델 로드가 안되는 문제가 발생했습니다. 저장은 되는데 로드가 안되는 문제가 남아있었던 것입니다. 제미나이가 말하기를 다음부터는 모델 전체가 아닌 weight만 저장하는것이 더 안정적이라고 합니다. 이번주 진행상황은 여기까지 입니다.





요약하자면
DeepAnat 모델을 학습시키는것과 저장하는것에 성공했으나 로드를 실패했습니다.
구버전의 tensorflow,keras를 사용할때는 numpy, h5py 등의 패키지를 삭제하고 다시 구버전으로 다시 깔아야 합니다.






###### 모르는 용어
> 피질,회백질,백질,섬유 대충 뇌 구조 관련
> 다이스 계수
> dMRI
> DWI
