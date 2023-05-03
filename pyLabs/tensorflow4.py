import tensorflow as tf
import numpy as np

X=np.array([[.05,.10]]) # 입력데이터 <=
YT=np.array([[0.,1.]]) # 크로스 엔트로피는 하나의 목표 값만 1, 나머지는 0
W=np.array([[.15,.25],[.20,.30]]) # 가중치 <=
B=np.array([.35,.35]) # 편향
W2=np.array([[.40,.50],[.45,.55]]) # 가중치 <=
B2=np.array([.60,.60]) # 편향

model=tf.keras.Sequential([
tf.keras.Input(shape=(2,)),
tf.keras.layers.Dense(2, activation='relu'),
tf.keras.layers.Dense(2, activation='softmax')
]) # 신경망 모양 결정(W, B 내부적 준비)

model.layers[0].set_weights([W,B])
model.layers[1].set_weights([W2,B2])

model.compile(optimizer='sgd',# 7공식, 학습
loss='categorical_crossentropy') # 2공식, 오차계산

Y=model.predict(X)
print(Y)

model.fit(X,YT,epochs=100000)

print('W=',model.layers[0].get_weights()[0])
print('B=',model.layers[0].get_weights()[1])
print('W2=',model.layers[1].get_weights()[0])
print('B2=',model.layers[1].get_weights()[1])

Y=model.predict(X)
print(Y)