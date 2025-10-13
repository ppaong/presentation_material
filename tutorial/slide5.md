# 진행척도
6장 완료




### transfer learning을 하는법
1. 모델 아무거나 하나를 불러온다.
2. 중간에 바꿀 부분의 in_features 속성을 얻는다.
3. 임의의 Layer를 만들고, 중간을 임의의 Layer로 교체한다.

model_ft = models.resnet18(weights = 'IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs,2)

모델이 아에 학습 못하게 grad를 막아버리는것도 해볼 수 있다.
for param in model_fc.parameters():
    param.requires_grad = False

(Layer교체를 이 이후에 하면 바꾼 Layer만 grad가 살아있게 된다.)



### Adversarial Example Generation
cnn이 잘 학습했는지 확인해보기 위해서 Fast Gradient Sign Attack이라는 것을 해볼 수 있다.
이미지에 약간에 (육안으로는 구별도 안되는)잔상 같은걸 주면 모델이 아에 다른 class로 구분하는 문제가 발생하는것을 확인해보는 것이다.
일종의 inference 단계에서 noise injection을 해서 accuracy가 노이즈 강도에 따라 얼마나 떨어지는지를 테스트 과정이라 볼수도 있을것 같다.
이런거 해결하려면 학습 단계부터 noise injection해서 해결보거나 데이터를 더 뻥튀기 해서 돌려보는것이 도움이 될 것이라 생각한다.

def fgsm_attack(image, eps, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + eps*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

이런 함수 하나를 선언해두고 test함수를 따로 만들어서 노이즈 정도에 따른 accuracy하락을 시각화하면 효과적으로 일반화 에러가 어느정도인지 파악할수있다.



### 가중치 초기값을 다음과 같이 초기화 할 수 있다.
layer 종류에 따라 각각 초기화하는 방식을 지정해줄 수 있다
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

net.apply(weights_init)



### 아래의 parallel에서 gpu를 병렬로 사용 가능하다.
좀 무거운 모델을 돌릴때 사용하면 좋을것 같다.
import torch.nn.parallel

net = Net().to(device)
net = nn.DataParallel(net, list(range(ngpu)))



### 한 장치에서 나오는 결과가 항상 같은 결과를 보이게 강제할 수 있다. (deterministic)
torch.use_deterministic_algorithms(True)



### GAN같이 2개의 모델로 학습하는 경우 각각의 optimizer를 따로 만들어 줄 필요가 있다.





