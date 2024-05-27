import tensorflow as tf
import numpy as np
import cv2
import os

# Path to data
SRC = 'cloth3d++_subset/'
print('SRC:', SRC)

N_TRAIN = 128
N_VAL = 16

subfolders = [x for x in os.listdir(SRC) if x.isdigit()]
subfolders.sort(key=int)
print(subfolders)

training_folders = subfolders[:N_TRAIN]
validation_folders = subfolders[N_TRAIN:N_TRAIN+N_VAL]
test_folders = subfolders[N_TRAIN+N_VAL:]

train_list = []
for sample in training_folders:
    try:
        train_list += os.listdir(SRC + sample + '/image')
    except:
        pass
train_list = list(map(lambda filename: filename[:-4], train_list))
to_remove = [261, 259, 188]

for x in to_remove:
    del train_list[x]

validation_list = []
for sample in validation_folders:
    try:
        validation_list += os.listdir(SRC + sample + '/image')
    except:
        pass
validation_list = list(map(lambda filename: filename[:-4], validation_list))

test_list = []
for sample in test_folders:
    try:
        test_list += os.listdir(SRC + sample + '/image')
    except:
        pass
test_list = list(map(lambda filename: filename[:-4], test_list))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image_path, depth_path):
    img = tf.io.decode_image(tf.io.read_file(image_path), channels=3)

    img_bytes = img.numpy().tobytes()

    depth = np.load(depth_path)
    depth = depth.astype(np.float32)
    depth_bytes = depth.tobytes()

    feature = {
        'image': _bytes_feature(img_bytes),
        'depth': _bytes_feature(depth_bytes),
        'height': _int64_feature(img.shape[0]),
        'width': _int64_feature(img.shape[1]),
        'depth_height': _int64_feature(depth.shape[0]),
        'depth_width': _int64_feature(depth.shape[1])
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def create_tfrecord(data_list, root_dir, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for file_name in data_list:
            image_path = os.path.join(root_dir, file_name[:5], 'image', file_name + '.jpg')
            depth_path = os.path.join(root_dir, file_name[:5], 'depth', file_name + '.npy')
            example = serialize_example(image_path, depth_path)
            writer.write(example)

if __name__ == '__main__':
    create_tfrecord(train_list, SRC, 'train.tfrecords')
    create_tfrecord(validation_list, SRC, 'validation.tfrecords')
    create_tfrecord(test_list, SRC, 'test.tfrecords')