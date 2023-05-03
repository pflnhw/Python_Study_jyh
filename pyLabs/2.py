import tensorflow as tf

mnist=tf.keras.datasets.mnist

(_,_),(x,yt)=mnist.load_data()
x=x/255
x=x.reshape((10000,784))

model = tf.keras.models.load_model('2_h')

import numpy as np
np.set_printoptions(precision=2,suppress=True)
y=model.predict(x[:1])
print(y[0])
print(yt[0])
print(np.argmax(y[0]))