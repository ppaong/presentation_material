이번주는 듀토리얼 7장 논문읽기 파트를 진행했습니다.
논문명은 Diffusion MRI data analysis assisted by deep learning synthesized anatomical images (DeepAnat)입니다.
https://www.sciencedirect.com/science/article/pii/S1361841523000051

모델 돌려보는 부분을 집중해서 공부했습니다.

### 논문 내용
먼저 제가 이해한 논문 내용은 다음과 같습니다.

Diffuction MRI 라는 데이터가 있습니다. neuroimaging, making the tissue microstructure and structural connections in the in vivo human brain하기위해 사용되는 데이터라고 합니다. 하지만 데이터를 분석하기 위해서는 추가적으로 t1w 뇌 스캔 이미지가 팔요한데 이것을 정확히 측정하는게(만들어내는게) 어려운 일입니다. 환자의 작은 움직임에도 오차가 발생하고 사진 찍는 시간도 굉장히 오래걸린다고 합니다. 그래서 논문 저자는 Diffusion MRI 데이터를 가지고 t1w이미지를 합성(생성)해보자고 제안합니다. 여기서 제안하는 모델 이름이 DeepAnat 이고 구현은 tensorflow 구버전으로 구현되어있습니다. CNN을 이용해서 dMRI로 부터 T1w를 만들어냅니다.     

DeepAnat모델의 구조는 일단 입력으로 복셀 이미지(3차원 이미지)(DiffusionMRI 데이터)를 여러장(meanb0, meandwi, dtiL1, dtiL2, dtiL3, dtiDwi1, dtiDwi2, dtiDwi3, dtiDwi4, dtiDwi5, dtiDwi6) 받아서, 복셀 이미지를 출력하는 output을 가졌습니다.3D Unet을 사용해서 위와 같은 모델을 설계했고, 학습방식은 GAN방식으로 구현되어있습니다.     

여기서 특이사항은 generator로 사용된 Unet 모델을 3D 이고 discriminator로 사용된 모델은 2D이미지를 받는다는 점입니다.    
저자는 SRGAN을 근거로 들며, GAN학습방식을 사용할때 discriminator는 2D가 좋다고 합니다.    

loss function은 다음과 같은 함수가 사용되었습니다.   
$$ L = L_{MAE} + \lambda * L_{adversarial} $$     
람다를 통해서 data fitting loss와 adversarial loss를 적절한 비율로 조정해서 loss를 사용했습니다.(람다는 10^-3 정도 라고 합니다.)     

논문 내용은 여기까지이고 github에 저자가 구현해둔 DeepAnat파일을 실행시켜본 내용은 다음과 같습니다.    



### 환경
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



### 환경설정 문제점
* 구버전의 tensorflow와 numpy 최신버전은 충돌이 일어나는것 같습니다. 때문에 numpy도 구버전으로 다시 깔았습니다.
* conda로 까는것과 pip으로 까는것은 약간의 차이가 있는것 같습니다. 가능하면 pip으로 까는것이 좋은것 같습니다. conda로 설치했을때는 충돌 나던것이 pip으로 설치하니 멈추었습니다.
* import 부분에서 계속 오류가 나서 콘다 환경을 삭제하고 패키지 다시깔기를 계속하며 얻어낸 결과인데, vscode 에서 주피터를 실행하여 커널을 찾을때는 vscode를 종료했다가 다시 시작하고 커넬을 찾는게 좋습니다. 잘못하면 과거에 삭제된 환경을 vscode에서 연결시킬수 있습니다.



### 학습 문제점
* 저장부분에서 계속 h5py패키지가 깔려있음에도 import하지 못하는 문제가 있어서 체크포인트마다 학습이 멈추는 문제가 있었습니다. tensorflow 구버전에서는 h5py를 삭제하고 다시 구버전으로 다시 깔아줘야 하는것 같습니다.
* 추가적으로 배치 사이즈가 작고, 학습 속도가 느려서 멀티 GPU로 돌려보려고 여러 차례 시도했으나, 구버전 cudnn파일의 문제인지 자꾸 No OpKernel was registered to support Op 'NcclAllReduce' 에러가 뜨며 어떤 연산이 누락되서 실행할 수 없었습니다. (8.0.5 ~ 7.6.0 , 7.5.0 버전의 cudnn을 가져와서 시도해보았으나 전부 같은 에러가 떴습니다. 도저히 고칠수 없어서 싱글 GPU를 사용하고 일단 넘어갔습니다.)



### 학습 결과
대부분 문제를 해결하고 학습을 돌려봤습니다.    
batchsize = 1    
epoch = 30     
기본 설정대로 진행하였으며 시간은 대략 2시간 정도 걸렸습니다.    

```
Train on 640 samples, validate on 96 samples
Epoch 1/30
640/640 [==============================] - 373s 583ms/sample - loss: 0.2268 - val_loss: 0.1739
Epoch 2/30
640/640 [==============================] - 380s 593ms/sample - loss: 0.1759 - val_loss: 0.1549
Epoch 3/30
640/640 [==============================] - 379s 593ms/sample - loss: 0.1629 - val_loss: 0.1482
Epoch 4/30
640/640 [==============================] - 379s 592ms/sample - loss: 0.1537 - val_loss: 0.1473
Epoch 5/30
640/640 [==============================] - 379s 593ms/sample - loss: 0.1467 - val_loss: 0.1415
Epoch 6/30
640/640 [==============================] - 379s 592ms/sample - loss: 0.1391 - val_loss: 0.1326
Epoch 7/30
640/640 [==============================] - 378s 591ms/sample - loss: 0.1342 - val_loss: 0.1342
Epoch 8/30
640/640 [==============================] - 380s 593ms/sample - loss: 0.1293 - val_loss: 0.1316
Epoch 9/30
640/640 [==============================] - 379s 592ms/sample - loss: 0.1247 - val_loss: 0.1268
Epoch 10/30
640/640 [==============================] - 379s 592ms/sample - loss: 0.1221 - val_loss: 0.1252
Epoch 11/30
640/640 [==============================] - 378s 591ms/sample - loss: 0.1180 - val_loss: 0.1262
Epoch 12/30
640/640 [==============================] - 378s 591ms/sample - loss: 0.1156 - val_loss: 0.1278
Epoch 13/30
640/640 [==============================] - 379s 592ms/sample - loss: 0.1135 - val_loss: 0.1242
Epoch 14/30
640/640 [==============================] - 378s 591ms/sample - loss: 0.1103 - val_loss: 0.1246
Epoch 15/30
640/640 [==============================] - 379s 592ms/sample - loss: 0.1095 - val_loss: 0.1252
Epoch 16/30
640/640 [==============================] - 380s 593ms/sample - loss: 0.1079 - val_loss: 0.1218
Epoch 17/30
640/640 [==============================] - 379s 592ms/sample - loss: 0.1049 - val_loss: 0.1220
Epoch 18/30
640/640 [==============================] - 379s 591ms/sample - loss: 0.1029 - val_loss: 0.1245
Epoch 19/30
640/640 [==============================] - 378s 591ms/sample - loss: 0.1015 - val_loss: 0.1247
Epoch 20/30
640/640 [==============================] - 378s 591ms/sample - loss: 0.1012 - val_loss: 0.1251
Epoch 21/30
640/640 [==============================] - 379s 591ms/sample - loss: 0.1000 - val_loss: 0.1228
Epoch 22/30
640/640 [==============================] - 379s 593ms/sample - loss: 0.0984 - val_loss: 0.1202
Epoch 23/30
640/640 [==============================] - 380s 594ms/sample - loss: 0.0961 - val_loss: 0.1194
Epoch 24/30
640/640 [==============================] - 379s 592ms/sample - loss: 0.0967 - val_loss: 0.1193
Epoch 25/30
640/640 [==============================] - 379s 592ms/sample - loss: 0.0951 - val_loss: 0.1201
Epoch 26/30
640/640 [==============================] - 379s 593ms/sample - loss: 0.0946 - val_loss: 0.1192
Epoch 27/30
640/640 [==============================] - 378s 591ms/sample - loss: 0.0947 - val_loss: 0.1196
Epoch 28/30
640/640 [==============================] - 379s 592ms/sample - loss: 0.0925 - val_loss: 0.1189
Epoch 29/30
640/640 [==============================] - 379s 592ms/sample - loss: 0.0940 - val_loss: 0.1180
Epoch 30/30
640/640 [==============================] - 378s 591ms/sample - loss: 0.0903 - val_loss: 0.1193
```

```
Applying...
mwu126426
mean squared error: 0.0010506010649782812
mwu127226
mean squared error: 0.0016210628910862254
mwu128026
mean squared error: 0.0010324606050715888
mwu128632
mean squared error: 0.0021544102031878856
mwu129937
mean squared error: 0.0012023388945916294
mwu130417
mean squared error: 0.0013345574308358452
Applying finished
```

다음날 loss그래프를 띄우고 모델에 데이터를 넣어서 샘플 이미지를 뽑아보고 싶었으나, 무슨 문제인지 모델 로드가 안되는 문제가 발생했습니다. 저장은 되는데 로드가 안되는 문제가 남아있었던 것입니다. 제미나이가 말하기를 다음부터는 모델 전체가 아닌 weight만 저장하는것이 더 안정적이라고 합니다. 이번주 진행상황은 여기까지 입니다.



### 요약
DeepAnat 모델을 학습시키는것과 저장하는것에 성공했으나 로드를 실패했습니다.    
구버전의 tensorflow,keras를 사용할때는 numpy, h5py 등의 패키지를 삭제하고 다시 구버전으로 다시 깔아야 합니다.   





