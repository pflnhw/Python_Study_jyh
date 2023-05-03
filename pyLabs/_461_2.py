import tensorflow as tf
from _7seg_data import X, YT

model=tf.keras.Sequential([
    tf.keras.Input(shape=(7,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(4, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')

model.fit(X,YT,epochs=10000)

Y=model.predict(X)
print(Y)

model.save('model.h5')