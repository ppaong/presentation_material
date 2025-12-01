# 진행척도
tutorial 8. PyTorch - DL for NLP -> 완료    
기초부터 시작하는 NLP: 문자-단위 RNN으로 이름 분류하기 - 완료    
기초부터 시작하는 NLP: 문자-단위 RNN으로 이름 생성하기 - 완료    
기초부터 시작하는 NLP: Sequence to Sequence 네트워크와 Attention을 이용한 번역 - 진행중

예제의 seq2seq 모델에 대해서 정리를 하는것이 좋을것 같아서 관련 내용을 정리했습니다.   

## seq2seq모델
seq를 받아서 seq를 생성하는 작업을 수행하는 모델입니다.    
토크나이징을 어떻게 하느냐에 따라, 글자 단위로 seq를 만들면 단어가 되고, 단어 단위로 seq를 만들면 문장으로 만들수 있습니다.     

###### 이미지 출처 : https://yjjo.tistory.com/35

---

![img](https://blog.kakaocdn.net/dna/cByx3E/btqDyNiSnP2/AAAAAAAAAAAAAAAAAAAAAGgHIPk9EsLAj0uU_KEmzINE7Pl8TT_Phcouc1ksidJr/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1767193199&allow_ip=&allow_referer=&signature=uirRDC5eygM2Dg22j42RnJzpxrg%3D)
seq2seq는 전체적으로 2개의 파트로 나뉩니다.(즉 2개의 RNN모델이 필요)    
encoder에서는 입력 seq를 익어 맥락과 내용을 잘 정리한 hidden state를 만듭니다.    
decoder에서는 encoder에서 받아온 hidden state로 부터 재귀적인 방식으로 문장을 생성합니다(<EOS>토큰까지)   
두 모델 전부 LSTM 이나 GRU 혹은 그냥 RNN을 사용합니다.    

---

![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FQQBuD%2FbtqDCaRbHNF%2FAAAAAAAAAAAAAAAAAAAAAGgO8VGwhHiN0b5yOXKYXSY84YxfDf_8L6n0vUfsVIzS%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3DmjahYe30%252BMF0%252BfyIKqdWaNKTC30%253D)


문장이나 단어를 seq로 전처리 할때 끝 부분에 <END> 토큰을 추가해서 하거나 혹은 모델 자체에서 처리하도록 만들 수 있습니다.    
pytorch 예제의 경우 토크나이징은 전부 one-hot으로 처리했으나, embedding 모델도 처리할 수 있습니다.   

---

![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FVGRNm%2FbtqDCb3D0Wn%2FAAAAAAAAAAAAAAAAAAAAAA3rr3vlfQsoQfu3vBTDUguO6E87r0xnZp7IaUHF8hx1%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3DKV2Tg9LGvwas2odbDQKO56NLkSg%253D)
디코더도 rnn이기 때문에 hidden state만으로는 작동 불가능하고 input 토큰이 있어야 시작하는데, 이 토큰을 문맥적으로 아무 의미 없는 <SOS> 넣어 시작합니다.
그 뒤로 재귀적으로 가며 문장 끝에서 <EOS> 토큰을 뽑아낼때까지 반복합니다.

```py
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)#SOS 토큰 시작
        decoder_hidden = encoder_hidden#받아온 hidden state
        decoder_outputs = []

        for i in range(MAX_LENGTH):#계속 반복될 수 있으니 최대길이를 정해줘야한다.
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)#한 step씩 구현
            decoder_outputs.append(decoder_output) 

            if target_tensor is not None:
                # Teacher forcing 포함: 목표를 다음 입력으로 전달
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Teacher forcing 미포함: 자신의 예측을 다음 입력으로 사용
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # 입력으로 사용할 부분을 히스토리에서 분리

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None   # 학습 루프의 일관성 유지를 위해 `None` 을 추가로 반환

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)#gru에서 한 단계
        output = self.out(output)
        return output, hidden
```



### 한계점  
너무 긴 문장은 잘 처리하지 못합니다. (RNN의 고질적 문제점)     
단점으로는 순차적으로 계산하며 hidden state를 넘겨야해서 병렬처리가 안 되는 이유로 학습이 매우 느립니다.   



## 바다나우 어텐션(Bahdanau Attention)
hidden state 하나에 문장을 거쳐오며 모든 정보를 다 압축시키는것은 한계점이 명확합니다.    
위의 한계점을 개선하고자 나온 seq2seq에서 attention 방식 입니다.    
부가적 어텐션(Additive Attention)로도 불립니다.

###### 이미지 출처 : https://yjjo.tistory.com/46

---

![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FcvArVO%2Fbtrm9E2GDC2%2FAAAAAAAAAAAAAAAAAAAAAHCprBUzCAxmWXRPP4JqYiO-oxhsr_e1XS4mPsP88avc%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3DYVulQELD5Uowx9uP%252FeILV4nP5tA%253D)
![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FcNcsQd%2Fbtrl4RHOz6b%2FAAAAAAAAAAAAAAAAAAAAALr-3q2Mz3WMQRXvchyA2J6Fg6JfLw_mzY2RFMuSxvkl%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3DDHZn77fhab%252BdsBvOeXWyjvZWABc%253D)


```py
class BahdanauAttention(nn.Module):#attention score 계산을 위한 모델
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)#softmax
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing 포함: 목표를 다음 입력으로 전달
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Teacher forcing 미포함: 자신의 예측을 다음 입력으로 사용
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # 입력으로 사용할 부분을 히스토리에서 분리

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
```

