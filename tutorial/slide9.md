# 진행척도
tutorial 8. PyTorch - DL for NLP
1. 기초부터 시작하는 NLP: 문자-단위 RNN으로 이름 분류하기 - 완료
2. 기초부터 시작하는 NLP: 문자-단위 RNN으로 이름 생성하기 - 진행
3. 기초부터 시작하는 NLP: Sequence to Sequence 네트워크와 Attention을 이용한 번역 - 미완료



# 기초부터 시작하는 NLP: 문자-단위 RNN으로 이름 분류하기

### 데이터 전처리
unicode 문자열을 단순히 ASCII로 전처리해서 임시로 데이터를 간단한 형식으로 다뤄볼 수 있다.
```py
#예제 코드
import string
import unicodedata

allowed_characters = string.ascii_letters + " .,;'" + "_" #ascii 일부 문자와 ".,;'" 만 허용 "_"는 공백문자로 활용
n_letters = len(allowed_characters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) #발음 기호와 문자를 분리시키는 작업. (유니코드는 같은 문자임에도 다른 코드를 지닐수도 있다.)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )
```
```py
import torch

def letterToIndex(letter):
    if letter not in allowed_characters:
        return allowed_characters.find("_")
    else:
        return allowed_characters.find(letter)

def letterToTensor(letter): #글자를 one-hot 인코딩
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line): #문자 단위로 BoW 표현하는 예시 코드. 글자는 고정된 문자들만 사용한다. 
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor
```
먼저 전처리 하는 메소드 부터 만들고 다음에 Dataset을 클래스를 만들어서 데이터를 다루면 편리하다. 구현은 기초부터 해야한다.


### 학습및 샘플링
전반적인 train 함수의 구현은 다음과 같이 한다.
```py
def train(rnn, training_data, n_epoch = 10, n_batch_size = 64, report_every = 50, learning_rate = 0.2, criterion = nn.NLLLoss()):
    #모델을 하나 학습 시켜도 함수처럼 만들어서 train을 구현하는것이 좋은것 같다. 확장 가능성이 크다.
    current_loss = 0 #로그 및 출력용 loss
    all_losses = []

    rnn.train() #모델 train 모드 설정
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate) #optimizer 정의

    start = time.time() #시간 기록용
    print(f"training on data set with n = {len(training_data)}")

    for iter in range(1, n_epoch + 1):
        rnn.zero_grad() #epoch마다 grad 초기화

        batches = list(range(len(training_data))) #minibatch 사용. Dataset에서 구현해서 사용해도 된다.
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) //n_batch_size )

        for idx, batch in enumerate(batches):
            batch_loss = 0 #minibatch loss
            for i in batch: #minibatch 사용. Dataset에서 구현해서 사용해도 된다.
                (label_tensor, text_tensor, label, text) = training_data[i]
                output = rnn.forward(text_tensor)
                loss = criterion(output, label_tensor) #loss 계산
                batch_loss += loss #loss 합산

            batch_loss.backward() #역전파 계산
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step() #step
            optimizer.zero_grad() #optimizer 초기화

            current_loss += batch_loss.item() / len(batch) #로그 기록

        all_losses.append(current_loss / len(batches) ) #로그 기록
        if iter % report_every == 0:
            print(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
        current_loss = 0

      return all_losses #loss및 기타 로그를 반환해서 데이터를 시각적으로 활용할 준비
```
evaluate 함수
```py
def evaluate(rnn, testing_data, classes):
    confusion = torch.zeros(len(classes), len(classes))

    rnn.eval() #모델 eval 모드 설정
    with torch.no_grad(): #grad 계산 안하고 기록도 안하게 하기(자원 절약)
        for i in range(len(testing_data)): #모델을 사용하는 코드
            (label_tensor, text_tensor, label, text) = testing_data[i]
            output = rnn(text_tensor)
            guess, guess_i = label_from_output(output, classes)
            label_i = classes.index(label)
            confusion[label_i][guess_i] += 1

    for i in range(len(classes)): #얻어낸 데이터로 나머지 처리하는 코드
        denom = confusion[i].sum()
        if denom > 0:
            confusion[i] = confusion[i] / denom

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.cpu().numpy())
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
```



# 실습 결과

# 추가
예제에서는 RNN만 사용했는데 pytorch는 LSTM과 GRU도 지원합니다. 학습이 더 잘되나 궁금해서 테스트 해보았습니다.





