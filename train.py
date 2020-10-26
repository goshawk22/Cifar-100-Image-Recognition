import numpy as np
from tensorflow.keras.datasets.cifar100 import load_data
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical

# Load data
(x_train, y_train), (x_test, y_test) = load_data(label_mode="fine")

classes = np.amax(y_train) + 1

# Normalize data

x_train = np.array(x_train, dtype=float) / 255.0
x_test = np.array(x_test, dtype=float) / 255.0

x_train = x_train.reshape([-1, 32, 32, 3])
x_test = x_test.reshape([-1, 32, 32, 3])

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

# Build model
input1 = Input(shape=(32,32,3))


conv1 = Conv2D(32, kernel_size=3, activation='relu')(input1)
drop1 = Dropout(0.25)(conv1)

conv2 = Conv2D(64, kernel_size=3, activation='relu')(drop1)
pad1 = ZeroPadding2D(padding=((0,1), (0,1)))(conv2)
bat1 = BatchNormalization(momentum=0.8)(pad1)
drop2 = Dropout(0.25)(bat1)

conv3 = Conv2D(128, kernel_size=3, activation='relu')(drop2)
pad2 = ZeroPadding2D(padding=((0,1), (0,1)))(conv3)
bat2 = BatchNormalization(momentum=0.8)(pad2)
drop3 = Dropout(0.25)(bat2)

conv3 = Conv2D(128, kernel_size=3, activation='relu')(drop2)
pad2 = ZeroPadding2D(padding=((0,1), (0,1)))(conv3)
bat2 = BatchNormalization(momentum=0.8)(pad2)
drop3 = Dropout(0.25)(bat2)

flat = Flatten()(drop3)
output = Dense(classes, activation='softmax')(flat)


model = Model(inputs=[input1], outputs=[output])

model.summary()

# Train
optimizer = Adam(learning_rate=0.001)

model.compile(loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=128, validation_split=0.2, shuffle=True)