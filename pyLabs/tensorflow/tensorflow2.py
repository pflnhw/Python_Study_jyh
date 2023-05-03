import tensorflow as tf

mnist=tf.keras.datasets.mnist

(X,YT),(x,yt)=mnist.load_data()
print(X.shape,YT.shape,x.shape,yt.shape)

from matplotlib import pyplot as plt
# plt.imshow(X[0])
# plt.show()
print(YT[0:3])
print(tf.one_hot(YT[0:3],10))

# for row in range(28):
#     for col in range(28):
#         print('%4s'%X[0][row][col],end='')
#     print()
    
import numpy as np


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X[i],cmap=plt.cm.binary)
    plt.xlabel(YT[i])
# plt.show()

X,x=X/255,x/255
X,x=X.reshape((60000,28,28,1)),x.reshape((10000,28,28,1))
print(X.shape)

model=tf.keras.Sequential([
    tf.keras.Input(shape=(28,28,1)),
    tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10,activation='softmax')])

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X,YT,epochs=5)

model.evaluate(x,yt)

model.save('model.h5')