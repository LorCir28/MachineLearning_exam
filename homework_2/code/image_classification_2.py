# ARCHITECTURE:
# the cnn has 4 convolutional layers, 4 avgpooling layers, 7 batchnormalization, 3 dropout and 4 dense (fully connected) layers
# dropout and batchnormalization are kind of regularization

# OPTIMIZER:
# the optimizer is SGD (standard stochastic  gradient descent) with learning rate 0.01
# the optimizer is the process that will execute the actual training

# REGULARIZER:
# the regularizer is l1 with value 0.0001

# HYPERPARAMETERS:
# filters = 16 and 32
# kernel_size = (5, 5)
# strides = (2, 2)
# activation = tanh
# dense_layers: 1000, 700, 300, 10

# PREPROCESSING:
# batch_size = 32
# validation_split = 0.2


import tensorflow as tf
import numpy as np
import sklearn.metrics 
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


base_dir = r'C:\Users\lollo\OneDrive\Desktop\università\magistrale\year_1\semester_1\machine_learning\homework_2\dataset'

# there are 4128 images totally

# preprocessing (load data)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(64, 64),
    batch_size=32,
    subset='training'
)

test_generator = test_datagen.flow_from_directory(
    base_dir,
    target_size=(64, 64),
    batch_size=32,
    subset='validation'
)

num_samples = train_generator.n
num_classes = train_generator.num_classes
input_shape = train_generator.image_shape

classnames = [k for k,v in train_generator.class_indices.items()]

print('\n\n')

print("Image input %s" %str(input_shape))
print("Classes: %r" %classnames)

print('Loaded %d training samples from %d classes.' %(num_samples,num_classes))
print('Loaded %d test samples from %d classes.' %(test_generator.n,test_generator.num_classes))


# cnn
cnn = tf.keras.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=16, padding='same', strides=(2, 2), kernel_size=(5, 5), activation='tanh', input_shape=input_shape))
cnn.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
cnn.add(tf.keras.layers.BatchNormalization())

cnn.add(tf.keras.layers.Conv2D(filters=16, padding='same', strides=(2, 2), kernel_size=(5, 5), activation='tanh'))
cnn.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
cnn.add(tf.keras.layers.BatchNormalization())

cnn.add(tf.keras.layers.Conv2D(filters=32, padding='same', strides=(1, 1), kernel_size=(5, 5), activation='tanh'))
cnn.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
cnn.add(tf.keras.layers.BatchNormalization())

cnn.add(tf.keras.layers.Conv2D(filters=32, padding='same', strides=(2, 2), kernel_size=(5, 5), activation='tanh'))
cnn.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
cnn.add(tf.keras.layers.BatchNormalization())

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(1000, activation='tanh', kernel_regularizer=tf.keras.regularizers.l1(0.0001)))
cnn.add(tf.keras.layers.Dropout(0.4))
cnn.add(tf.keras.layers.BatchNormalization())

cnn.add(tf.keras.layers.Dense(700, activation='tanh', kernel_regularizer=tf.keras.regularizers.l1(0.0001)))
cnn.add(tf.keras.layers.Dropout(0.4))
cnn.add(tf.keras.layers.BatchNormalization())

cnn.add(tf.keras.layers.Dense(300, activation='tanh', kernel_regularizer=tf.keras.regularizers.l1(0.0001)))
cnn.add(tf.keras.layers.Dropout(0.4))
cnn.add(tf.keras.layers.BatchNormalization())

cnn.add(tf.keras.layers.Dense(10, activation='softmax'))

cnn.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
cnn.summary()


# train
hist = cnn.fit(train_generator, epochs=20, validation_data=test_generator)


# evaluate the model
loss, acc = cnn.evaluate(test_generator)
print('Test loss: %f' %loss)
print('Test accuracy: %f' %acc)

# precision, recall, F-score
# test_generator = test_datagen.flow_from_directory(
#     base_dir,
#     target_size=(64, 64),
#     batch_size=32,
#     subset='validation'
# )

preds = cnn.predict(test_generator)

Ypred = np.argmax(preds, axis=1)
Ytest = test_generator.classes

print(classification_report(Ytest, Ypred, labels=None, target_names=classnames, digits=3))


# confusion matrix
cm = confusion_matrix(Ytest, Ypred)

conf = []
for i in range(0,cm.shape[0]):
  for j in range(0,cm.shape[1]):
    if (i!=j and cm[i][j]>0):
      conf.append([i,j,cm[i][j]])

col=2
conf = np.array(conf)
conf = conf[np.argsort(-conf[:,col])]

print('%-16s     %-16s  \t%s \t%s ' %('True','Predicted','errors','err %'))
print('------------------------------------------------------------------')
for k in conf:
  print('%-16s ->  %-16s  \t%d \t%.2f %% ' %(classnames[k[0]],classnames[k[1]],k[2],k[2]*100.0/test_generator.n))


# plot results

# summarize history for accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()