#

## seq2seq모델
seq를 받아서 seq를 생성하는 작업이다.
보통 seq2seq모델의 문장 입력과 출력 방식은 다음과 같이 이루어진다.
![img](https://blog.kakaocdn.net/dna/cByx3E/btqDyNiSnP2/AAAAAAAAAAAAAAAAAAAAAGgHIPk9EsLAj0uU_KEmzINE7Pl8TT_Phcouc1ksidJr/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&expires=1767193199&allow_ip=&allow_referer=&signature=uirRDC5eygM2Dg22j42RnJzpxrg%3D)
![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FQQBuD%2FbtqDCaRbHNF%2FAAAAAAAAAAAAAAAAAAAAAGgO8VGwhHiN0b5yOXKYXSY84YxfDf_8L6n0vUfsVIzS%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3DmjahYe30%252BMF0%252BfyIKqdWaNKTC30%253D)
![img](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FVGRNm%2FbtqDCb3D0Wn%2FAAAAAAAAAAAAAAAAAAAAAA3rr3vlfQsoQfu3vBTDUguO6E87r0xnZp7IaUHF8hx1%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3DKV2Tg9LGvwas2odbDQKO56NLkSg%253D)
###### 이미지 출처: https://yjjo.tistory.com/35
* 디코더에서 문장 시작과 끝의 토큰을 <EOS>
*

RNN 계열의 LSTM 이나 GRU를 사용하며 
단점으로는 순차적으로 계산하며 hidden state를 넘겨야해서 병렬처리가 안 되고, 
###### 이제는 transformer에게 밀려버린 범부여..



## 바다나우 어텐션(Bahdanau Attention)
부가적 어텐션(Additive Attention)로도 불린다. 
