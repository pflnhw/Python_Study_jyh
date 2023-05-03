import tensorflow as tf

mnist=tf.keras.datasets.mnist

(_,_),(x,yt)=mnist.load_data()

x=x/255
x=x.reshape((10000,784))

model=tf.keras.models.load_model('model.h5')

y=model.predict(x)
print(y[0],yt[0])

import numpy as np
print(np.argmax(y[0]))

import matplotlib.pyplot as plt

x=x.reshape(10000,28,28)
# plt.imshow(x[0])
# plt.show()

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x[i],cmap=plt.cm.binary)
#     plt.xlabel(np.argmax(y[i]))
# plt.show()

cnt_wrong=0
y_wrong=[]
for i in range(10000):
    if np.argmax(y[i])!=yt[i]:
        y_wrong.append(i)
        cnt_wrong+=1
print(cnt_wrong)
print(y_wrong[:10])

# plt.imshow(x[247])
# plt.show()
# print(np.argmax(y[247]),yt[247])

plt.figure(figsize=(10,10))
for i in range(25,50):
    plt.subplot(5,5,i+1-25)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x[y_wrong[i]],cmap=plt.cm.binary)
    plt.xlabel(f'{y_wrong[i]}:y{np.argmax(y[y_wrong[i]])} yt{yt[y_wrong[i]]}')
plt.show()