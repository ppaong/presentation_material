# 진행척도
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
디코더에서 문장 시작과 끝의 토큰을 <EOS>
decoder가 올바른 원래문장을 생성하기 위해서는 모든 시점에서 이전단계의 결과물이 
이런 방식으로 decoder가 학습하는 방식을 teacher forcing 이라 부른다.



### 한계점  
너무 긴 문장은 잘 처리하지 못합니다. (RNN의 고질적 문제점)   
단점으로는 순차적으로 계산하며 hidden state를 넘겨야해서 병렬처리가 안 되는 이유로 학습이 매우 느립니다.
###### 이제는 transformer에게 밀려버린 범부여..



## 바다나우 어텐션(Bahdanau Attention)
hidden state 하나에 문장을 거쳐오며 모든 정보를 다 압축시키는것은 한계점이 명확합니다.
위의 한계점을 개선하고자 나온 seq2seq에서 attention 방식 입니다.
부가적 어텐션(Additive Attention)로도 불립니다.

###### 이미지 출처 : https://yjjo.tistory.com/46

---

![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FcvArVO%2Fbtrm9E2GDC2%2FAAAAAAAAAAAAAAAAAAAAAHCprBUzCAxmWXRPP4JqYiO-oxhsr_e1XS4mPsP88avc%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3DYVulQELD5Uowx9uP%252FeILV4nP5tA%253D)
![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FcNcsQd%2Fbtrl4RHOz6b%2FAAAAAAAAAAAAAAAAAAAAALr-3q2Mz3WMQRXvchyA2J6Fg6JfLw_mzY2RFMuSxvkl%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3DDHZn77fhab%252BdsBvOeXWyjvZWABc%253D)




