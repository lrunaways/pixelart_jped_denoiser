import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing as mp
import time
from random import shuffle
from GAN_pixelation.ResUnet import ResUnet
from tensorflow.keras.optimizers import Adam

N_EPOCHS = 3
STEPS_PER_EPOCH = 1024

TFRECORDS_DIRECTORY_PATH = r'C:\Users\Nikita\PycharmProjects\pixelart_jpeg_artifacts_removal\data\TFRecords_data'
feature_description = {
    'patch_compressed': tf.io.FixedLenFeature([], tf.string),
    'patch_clean': tf.io.FixedLenFeature([], tf.string),
}

def parse(x):
    x = tf.io.parse_single_example(x, feature_description)
    x = tf.stack([tf.io.parse_tensor(x['patch_compressed'], tf.dtypes.uint8),\
                    tf.io.parse_tensor(x['patch_clean'], tf.dtypes.uint8)])
    x = tf.dtypes.cast(x, tf.float32)
    x /= 255.0
    return [x[:,0], x[:, 1]]

# def preprocess(x):
#     x /= 255.0
#     return x

if __name__=="__main__":
    feature_description = {
        'patch_compressed': tf.io.FixedLenFeature([], tf.string),
        'patch_clean': tf.io.FixedLenFeature([], tf.string),
    }
    filenames = [os.path.join(TFRECORDS_DIRECTORY_PATH, fn) for fn in next(os.walk(TFRECORDS_DIRECTORY_PATH))[2]]
    train_dataset = tf.data.TFRecordDataset(filenames[:-1], compression_type='ZLIB')
    train_dataset = (train_dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                                  # .shuffle(3000, reshuffle_each_iteration=True)
                                  # .repeat()
                                  # .cache()
                                  .batch(2)
                                  # .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                                  # .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                   )
    val_dataset = tf.data.TFRecordDataset(filenames[-1], compression_type='ZLIB')
    val_dataset = val_dataset.take(2048)\
        .map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        # .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # val_dataset = (val_dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    #                           .repeat()
    #                           .batch(2)
    #                           .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #                           .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #                )
    model = ResUnet(4)
    model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error')
    # history = model.fit(train_dataset,
    #                               steps_per_epoch=1000,
    #                               epochs=N_EPOCHS,
    #                               validation_data=val_dataset,
    #                               validation_steps=2048,
    # )

    #
    #
    #
    print("!")
