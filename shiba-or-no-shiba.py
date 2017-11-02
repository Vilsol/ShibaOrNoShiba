import numpy as np

np.random.seed(1337)

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, SeparableConv2D
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.applications import Xception
from keras.applications.xception import preprocess_input
from keras.callbacks import Callback

from aiohttp import web

import tensorflow as tf

import os
import glob
import sys
import cv2
import json
import random

batch_size = 80
num_classes = 3
epochs = 1000


def get_images(directory, loss=0):
  images = []

  for i, file in enumerate(glob.iglob(directory)):
    if loss == 0 or random.random() > loss:
      images.append(preprocess_input(img_to_array(load_img(file))))

  return images

def load_dataset():
  shiba = 'shiba-or-no-shiba/shiba/*'
  doggo = 'shiba-or-no-shiba/doggo/*'
  random = 'shiba-or-no-shiba/random/*'

  dataset = []

  for i in get_images(shiba):
    dataset.append(
      (
        i,
        [1, 0, 0]
      )
    )

  for i in get_images(doggo):
    dataset.append(
      (
        i,
        [0, 1, 0]
      )
    )

  for i in get_images(random, 0.5):
    dataset.append(
      (
        i,
        [0, 0, 1]
      )
    )

  np.random.shuffle(dataset)

  x_train = []
  y_train = []

  x_test = []
  y_test = []

  for i in dataset:
    if np.random.random_sample() > 0.2:
      x_train.append(i[0])
      y_train.append(i[1])
    else:
      x_test.append(i[0])
      y_test.append(i[1])

  del dataset

  print("E")

  x_train = np.array(x_train)
  y_train = np.array(y_train)

  x_test = np.array(x_test)
  y_test = np.array(y_test)

  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')

  return x_train, x_test, y_train, y_test

class MyModelCheckpoint(Callback):
    def __init__(self, filepath, save_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MyModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.save_model = save_model

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.save_model.save_weights(filepath, overwrite=True)
                        else:
                            self.save_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.save_model.save_weights(filepath, overwrite=True)
                else:
                    self.save_model.save(filepath, overwrite=True)

def create_model(learn_rate=0.005):
  from keras.utils import multi_gpu_model
  with tf.device('/cpu:0'):
      # Alternative Xception model
      # model = Xception(weights=None,
      #                  input_shape=x_train.shape[1:],
      #                  classes=num_classes)

      img_input = Input(shape=(299, 299, 3))

      x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
      x = BatchNormalization(name='block1_conv1_bn')(x)
      x = Activation('relu', name='block1_conv1_act')(x)
      x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
      x = BatchNormalization(name='block1_conv2_bn')(x)
      x = Activation('relu', name='block1_conv2_act')(x)

      x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
      x = BatchNormalization(name='block13_sepconv1_bn')(x)
      x = Activation('relu', name='block13_sepconv2_act')(x)
      x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
      x = BatchNormalization(name='block13_sepconv2_bn')(x)

      x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)

      x = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
      x = BatchNormalization(name='block14_sepconv1_bn')(x)
      x = Activation('relu', name='block14_sepconv1_act')(x)

      x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
      x = BatchNormalization(name='block14_sepconv2_bn')(x)
      x = Activation('relu', name='block14_sepconv2_act')(x)

      x = GlobalAveragePooling2D(name='avg_pool')(x)
      x = Dense(num_classes, activation='softmax', name='predictions')(x)

      model = Model(inputs=[img_input], outputs=[x])

  parallel_model = multi_gpu_model(model, 4)

  parallel_model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
  return model, parallel_model

def train(x_train, x_test, y_train, y_test):
  model, parallel_model = create_model()

  model.summary()

  print("Preparing callbacks")

  board = keras.callbacks.TensorBoard()
  checkpoint = MyModelCheckpoint(filepath='./models/v12.{epoch:03d}-{val_loss:.4f}.hdf5', save_model=model)
  reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=0.00001, verbose=1)

  datagen = ImageDataGenerator(
      featurewise_center=False,
      samplewise_center=False,
      featurewise_std_normalization=False,
      samplewise_std_normalization=False,
      zca_whitening=False,
      rotation_range=0,
      width_shift_range=0.1,
      height_shift_range=0.1,
      horizontal_flip=True,
      vertical_flip=False)

  print("Fitting data generatror")
  datagen.fit(x_train)

  print("Training")
  parallel_model.fit_generator(datagen.flow(x_train, y_train,
                                   batch_size=batch_size),
                      steps_per_epoch=x_train.shape[0] // batch_size,
                      epochs=epochs,
                      validation_data=(x_test, y_test),
                      workers=4,
                      callbacks=[board, checkpoint, reduce_lr])

def resizeAndPad(img, size, padColor=127):
  h, w = img.shape[:2]
  sh, sw = size

  if h > sh or w > sw:
    interp = cv2.INTER_AREA
  else:
    interp = cv2.INTER_CUBIC

  aspect = w/h

  if aspect > 1:
    new_w = sw
    new_h = np.round(new_w/aspect).astype(int)
    pad_vert = (sh-new_h)/2
    pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
    pad_left, pad_right = 0, 0
  elif aspect < 1:
    new_h = sh
    new_w = np.round(new_h*aspect).astype(int)
    pad_horz = (sw-new_w)/2
    pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
    pad_top, pad_bot = 0, 0
  else:
    new_h, new_w = sh, sw
    pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

  if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)):
    padColor = [padColor]*3

  scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
  scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

  return scaled_img

def serve():
  models = {
    "xception": load_model("v11.198-0.0976.hdf5"),
    "custom": load_model("v12.537-0.1054.hdf5")
  }

  PORT = 8009

  if len(sys.argv) >= 4:
    PORT = int(sys.argv[3])

  async def handleGet(request):
    form = '<form method="post" enctype="multipart/form-data"><input type="file" name="image" accept="image/jpeg" required/><input type="submit"></form>'
    return web.Response(body="Shiba, Doggo, Random classifier<br>Likely very incorrect<br>Please POST a JPEG to this URL as the 'image' form field or use this form:<br><br>" + form, content_type='text/html')

  async def handlePost(request):
    try:
      data = await request.post()
      image = data['image']
      image = img_to_array(load_img(image.file))
      image = resizeAndPad(image, (299, 299))
      image = preprocess_input(np.array(image))

      results = {}

      for name, model in models.items():
        prediction = model.predict(np.array([image]))

        results[name] = {
          "shiba": float(prediction[0][0]),
          "doggo": float(prediction[0][1]),
          "random": float(prediction[0][2])
        }

      print("Prediction:", results)

      response = {
        "success": True,
        "predictions": results
      }

      return web.Response(body=json.dumps(response, sort_keys=True, indent=4).encode('utf-8'), content_type='application/json')
    except Exception as e:
      response = {
        "success": False,
        "error": str(e)
      }

      return web.Response(body=json.dumps(response, sort_keys=True, indent=4).encode('utf-8'), content_type='application/json')

  app = web.Application(client_max_size=4096**2)
  app.router.add_get('/', handleGet)
  app.router.add_post('/', handlePost)

  web.run_app(app, port=PORT)

if __name__ == '__main__':
  if len(sys.argv) == 1:
    print("Please enter command train or serve")
  else:
    if sys.argv[1] == "train":
      x_train, x_test, y_train, y_test = load_dataset()
      train(x_train, x_test, y_train, y_test)
    elif sys.argv[1] == "serve":
      serve()