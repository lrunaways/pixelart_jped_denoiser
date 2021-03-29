import os
import cv2
import numpy as np
import PIL
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import tensorflow as tf

MODEL_PATH = r"D:\thispanelkadoesnotexist\data\pixelart\pixel_size_finder\GAP_8ch_02dp.h5"

IN_RAW_PATH = r"D:\thispanelkadoesnotexist\data\pixelart\good_quality"
OUT_PIXELATED_PATH = r"D:\thispanelkadoesnotexist\data\pixelart\good_quality_1px\{}.png"


def find_shift(image, pixelation, reverse=False, swapaxes=()):
    min_shift_mse = 2 << 30
    best_shift = 0
    for shift in range(pixelation):
        im = image[:, shift:]
        if reverse:
            im = im[:,::-1]
        if swapaxes:
            im = np.swapaxes(im, swapaxes[0], swapaxes[1])
        im_res = cv2.resize(im, (int(im.shape[1] // pixelation),
                                 int(im.shape[0] // pixelation)),
                            interpolation=cv2.INTER_NEAREST)
        im_res = cv2.resize(im_res, (int(im.shape[1]),
                                     int(im.shape[0])),
                            interpolation=cv2.INTER_NEAREST)
        mse = np.sum((im - im_res) ** 2)
        if mse < min_shift_mse:
            min_shift_mse = mse
            best_shift = shift
    return min_shift_mse, best_shift

def create_model_GAP(nf=64, dp=0.25):
  model = tf.keras.models.Sequential()
  # model.add(tf.keras.layers.Lambda(lambda x: x[:8*(x.shape[1]//8 - 1), :8*(x.shape[2]//8 - 1)]))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2D(nf, (3, 3), padding='same', activation='elu'))
  # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
  model.add(tf.keras.layers.Dropout(dp))

  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2D(nf, (3, 3), padding='same', activation='elu'))
  # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(tf.keras.layers.Dropout(dp))

  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2D(nf, (3, 3), padding='same', activation='elu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
  model.add(tf.keras.layers.Dropout(dp))

  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2D(nf*2, (3, 3), padding='same', activation='elu'))
  # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
  model.add(tf.keras.layers.Dropout(dp))

  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2D(nf*2, (3, 3), padding='same', activation='elu'))
  # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(tf.keras.layers.Dropout(dp))

  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2D(nf*2, (3, 3), padding='same', activation='elu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
  model.add(tf.keras.layers.Dropout(dp))

  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2D(nf*4, (3, 3), padding='same', activation='elu'))
  # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
  model.add(tf.keras.layers.Dropout(dp))

  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2D(nf*4, (3, 3), padding='same', activation='elu'))
  # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(tf.keras.layers.Dropout(dp))

  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2D(nf*4, (3, 3), padding='same', activation='elu'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
  # model.add(tf.keras.layers.Dropout(dp))

  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Conv2D(8, (1, 1), padding='same', activation='elu'))
  model.add(tf.keras.layers.GlobalAveragePooling2D())

  # model.add(tf.keras.layers.Flatten())
  # model.add(tf.keras.layers.Dense(256))
  # model.add(tf.keras.layers.Activation('elu'))
  # model.add(tf.keras.layers.Dropout(0.5))
  model.add(tf.keras.layers.Dense(8))
  model.add(tf.keras.layers.Activation('softmax'))
  return model


if __name__=='__main__':
  if not os.path.exists(OUT_PIXELATED_PATH[:-6]):
      os.makedirs(OUT_PIXELATED_PATH[:-6])
  filenames = [os.path.join(IN_RAW_PATH, fn) for fn in next(os.walk(IN_RAW_PATH))[2]]
  print('Found {} files\n\n'.format(len(filenames)))

  model = create_model_GAP(8, 0.2)
  model.predict(np.zeros((1, 96, 96, 1)))
  model.load_weights(MODEL_PATH)

  for i, filename in enumerate(filenames[:]):
      try:
          print("{:>4}/{:<4} processing file '{}'".format(i, len(filenames), filename))
          image_raw = np.array(Image.open(filename.replace('\\', '/')).convert('RGB'))
          image = image_raw.copy()

          x_shift_beg, x_shift_end = 0, -1
          y_shift_beg, y_shift_end = 0, -1
          pred_label = -1

          # Привести к одному размеру пикселя
          while pred_label != 1:
              image_cut = image[max(0, image.shape[0]//2 - 256): image.shape[0]//2 + 256,
                                max(0, image.shape[1]//2 - 256): image.shape[1]//2 + 256]
              image_cut = image_cut[np.newaxis, :(image_cut.shape[0] // 8) * 8, :(image_cut.shape[1] // 8) * 8, 0:1]
              image_cut = (image_cut-image_cut.mean())/image_cut.std()
              pred = model.predict(image_cut)
              pred_label = pred.argmax()
              print('Predicted label: ', pred_label)
              x_shift_beg, x_shift_end = 0, -1
              y_shift_beg, y_shift_end = 0, -1
              if pred_label != 1:
                  min_mse = 2 << 30
                  pixelation = None
                  for k in [2, 3, 5, 7]:
                      im_res = cv2.resize(image, (int(image.shape[1] // k),
                                                  int(image.shape[0] // k)),
                                          interpolation=cv2.INTER_NEAREST)
                      im_res = cv2.resize(im_res, (int(image.shape[1]),
                                                   int(image.shape[0])),
                                          interpolation=cv2.INTER_NEAREST)
                      mse = np.sum((image - im_res) ** 2)
                      print(f"px {k} - mse: {mse}")
                      if mse < min_mse:
                          min_mse = mse
                          pixelation = k
                  if pixelation != pred_label:
                      print(f"Pred_label is {pred_label}, but {pixelation} with mse method!")

                  # Find shift to align image and pixel grid
                  mse, shift = find_shift(image,
                                          pred_label)
                  if mse < min_mse:
                      min_mse = mse
                      x_shift_beg = shift

                  mse, shift = find_shift(image[:, x_shift_beg:][:, ::-1],
                                          pred_label, reverse=True)
                  if mse < min_mse:
                      min_mse = mse
                      x_shift_end -= shift

                  mse, shift = find_shift(np.swapaxes(image[:, x_shift_beg: x_shift_end], 0, 1),
                                          pred_label, swapaxes=(0, 1))
                  if mse < min_mse:
                      min_mse = mse
                      y_shift_beg = shift

                  mse, shift = find_shift(np.swapaxes(image[y_shift_beg:, x_shift_beg: x_shift_end][::-1, :], 0, 1),
                                          pred_label, reverse=True, swapaxes=(0, 1))
                  if mse < min_mse:
                      min_mse = mse
                      y_shift_end -= shift

                  print(f'MSE: {min_mse}')
                  print('shifts:', x_shift_beg, x_shift_end, y_shift_beg, y_shift_end)
                  image = image[x_shift_beg:x_shift_end, y_shift_beg:y_shift_end]
                  image = cv2.resize(image, (int(image.shape[1] // pred_label),
                                             int(image.shape[0] // pred_label)),
                                               interpolation=cv2.INTER_NEAREST)
          # image = cv2.resize(image, (int(image.shape[1] * 2),
          #                            int(image.shape[0] * 2)),
          #                              interpolation=cv2.INTER_NEAREST)

          cv2.imwrite(OUT_PIXELATED_PATH.format(filename.split('\\')[-1].split('.')[0]),
                      cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      except PIL.UnidentifiedImageError as e:
          print("BAD FILE: ", e)

  print("Success!")
