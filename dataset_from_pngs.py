"""
Creates train and val data for pixelart jpeg artifacts removal from lossy images in IMAGES_DIRECTORY_PATH.
***Pixel brush size in every image must be 1! Convert with change_pixel_size.py***

Every image in a directory is compressed with jpeg artifacts and cut to patches.
Patches are written into x_train.npy/x_val.npy with corresponding patches without noise y_train.npy/y_val.npy
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing as mp
import time
from random import shuffle

IMAGES_DIRECTORY_PATH = r"D:\thispanelkadoesnotexist\data\pixelart\good_quality_1px"
OUT_DATASET_PATH = r'C:\Users\Nikita\PycharmProjects\pixelart_jpeg_artifacts_removal\data\TFRecords_data'

PATCH_SHAPE = (256, 256)
OS = 0

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def _int64_feature(value):
#   """Returns an int64_list from a bool / enum / int / uint."""
#   return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_example_TFRecord(x_image, y_image, writer):
    x_image = tf.convert_to_tensor(x_image, tf.dtypes.uint8)
    y_image = tf.convert_to_tensor(y_image, tf.dtypes.uint8)
    features = {
        # 'width': _int64_feature(x_image.shape[0]),
        # 'height': _int64_feature(x_image.shape[1]),
        # 'depth': _int64_feature(x_image.shape[2]),
        'patch_compressed': _bytes_feature(tf.io.serialize_tensor(x_image)),
        'patch_clean': _bytes_feature(tf.io.serialize_tensor(y_image))
    }

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())

    # image_feature_description = {
    #     'height': tf.io.FixedLenFeature([], tf.int64),
    #     'width': tf.io.FixedLenFeature([], tf.int64),
    #     'depth': tf.io.FixedLenFeature([], tf.int64),
    #     'patch_compressed': tf.io.FixedLenFeature([], tf.string),
    #     'patch_clean': tf.io.FixedLenFeature([], tf.string),
    # }
    # raw_image_dataset = tf.data.TFRecordDataset(r'C:\Users\Nikita\PycharmProjects\pixelart_jpeg_artifacts_removal\data\TFRecords_data\test.tfrecord')
    # # plt.imshow(tf.io.parse_tensor(list(parsed_image_dataset)[0]['patch_compressed'], tf.float32))
    # def _parse_image_function(example_proto):
    #     # Parse the input tf.train.Example proto using the dictionary above.
    #     return tf.io.parse_single_example(example_proto, image_feature_description)
    # parsed_image_dataset = raw_image_dataset.map(_parse_image_function)


def compress_image(image, max_compression, i):
    total_compression = 0
    filename = f'temp-{i}.jpg'
    while total_compression <= max_compression:
        compression = np.random.randint(0, max_compression - total_compression + 1)
        total_compression += compression
        cv2.imwrite(filename, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100 - compression])
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if np.random.random() < 0.8:
            break
    os.remove(filename)
    return image

def make_dataset(filenames):
    process_id = mp.current_process().name.split('-')[-1]
    with tf.io.TFRecordWriter(os.path.join(OUT_DATASET_PATH, f'data-{process_id}.tfrecord'),
                              tf.io.TFRecordOptions(compression_type='ZLIB')) as writer:
        for i, filename in enumerate(filenames):
            try:
                print("{:>4}/{:<4} processing file '{}'".format(i, len(filenames), filename))
                image_clean = cv2.imread(filename.replace('\\', '/'), cv2.IMREAD_COLOR)
                if image_clean is None:
                    print(filename)
                # randomly change pixel size to 2..8
                # do not use px size 1 as I've never met 1 px sized jpg
                pixel_sizes = set(np.arange(2, 8))

                for _ in range(3):  # make few iterations with random pixel_size for each image
                    # Compensate larger amount of patches from large pixel sized images
                    px_size_prob = np.arange(1, len(pixel_sizes) + 1)[::-1]
                    px_size_prob = px_size_prob / px_size_prob.sum()
                    new_pixel_size = np.random.choice(list(pixel_sizes), p=px_size_prob)
                    pixel_sizes -= set([new_pixel_size])
                    image_clean_resized = cv2.resize(image_clean, (int(image_clean.shape[1] * new_pixel_size),
                                                                   int(image_clean.shape[0] * new_pixel_size)),
                                                     interpolation=cv2.INTER_NEAREST)
                    if (image_clean_resized.shape[0] < PATCH_SHAPE[0] or
                            image_clean_resized.shape[1] < PATCH_SHAPE[1]):
                        print(f'Patch {PATCH_SHAPE} is larger than image {image_clean_resized.shape[:-1]}!')
                        continue

                    # add compression noise for most of images
                    if np.random.random() < 0.95:
                        compression = 0 if np.random.random() < 0.5 else int(
                            np.random.random() * 10 * new_pixel_size) // 2
                        image_compressed = compress_image(image_clean_resized, compression, process_id)
                    else:
                        image_compressed = image_clean_resized

                    x_steps, y_steps = np.arange(0, image_clean_resized.shape[1], PATCH_SHAPE[1]), \
                                       np.arange(0, image_clean_resized.shape[0], PATCH_SHAPE[0])
                    x_steps = (x_steps - OS).clip(0)
                    y_steps = (y_steps - OS).clip(0)
                    x_steps[-1], y_steps[-1] = image_clean_resized.shape[1] - PATCH_SHAPE[1], \
                                               image_clean_resized.shape[0] - PATCH_SHAPE[0]
                    patch_idx = 0
                    for x_crd in x_steps:
                        for y_crd in y_steps:
                            compressed_patch = image_compressed[y_crd:y_crd + PATCH_SHAPE[1],
                                                                x_crd:x_crd + PATCH_SHAPE[0]]
                            clean_patch = image_clean_resized[y_crd:y_crd + PATCH_SHAPE[1],
                                                              x_crd:x_crd + PATCH_SHAPE[0]]
                            write_example_TFRecord(compressed_patch, clean_patch, writer)
                            patch_idx += 1


            except TypeError as e:
                print("WRONG TYPE!")
                print(e)

if __name__=='__main__':
    if not os.path.exists(OUT_DATASET_PATH):
        os.makedirs(OUT_DATASET_PATH)
    filenames = [os.path.join(IMAGES_DIRECTORY_PATH, fn) for fn in next(os.walk(IMAGES_DIRECTORY_PATH))[2]]
    shuffle(filenames)
    filenames = filenames
    print('Found {} files\n\n'.format(len(filenames)))

    # with mp.Pool(32) as p:
    #    p.map(make_dataset, [np.arange(10)]*32)
    # make_dataset(filenames)
    processes = []  # a simple list to hold our process references
    n_workers = 5
    chunk_size = len(filenames)//n_workers
    for i in range(n_workers):
        work_set = filenames[i*chunk_size:(i+1)*chunk_size]
        process = mp.Process(target=make_dataset, args=(work_set, ))
        process.start()
        processes.append(process)
    results = [process.join() for process in processes]  # get the data back

    print("!")


