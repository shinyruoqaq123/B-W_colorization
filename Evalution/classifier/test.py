import tensorflow as tf
import os
import sys

batch_size = 1

data_dir = sys.argv[1]
test_cats_dir = data_dir + 'cats/'
test_dogs_dir = data_dir + 'dogs/'
test_panda_dir=data_dir+'panda/'


def _decode_resize(filename, label):
    image_string = tf.io.read_file(filename)
    image_docoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize(image_docoded, [256, 256]) / 255.0
    return image_resized, label


if __name__ == '__main__':
    test_cat_filenames = tf.constant([test_cats_dir + filename for filename in os.listdir(test_cats_dir)])
    test_dog_filenames = tf.constant([test_dogs_dir + filename for filename in os.listdir(test_dogs_dir)])
    test_panda_filenames = tf.constant([test_panda_dir + filename for filename in os.listdir(test_panda_dir)])
    test_filenames = tf.concat([test_cat_filenames, test_dog_filenames,test_panda_filenames], axis=-1)
    # 0:cat, 1:dog 2:panda
    test_labels = tf.concat([tf.zeros(test_cat_filenames.shape, dtype=tf.int32),
                             tf.ones(test_dog_filenames.shape, dtype=tf.int32)], axis=-1)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
    test_dataset = test_dataset.map(map_func=_decode_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size)

    model=tf.keras.models.load_model('../../checkpoint/classifier/model.h5')

    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for images,labels in test_dataset:
        y_pred = model.predict(images)
        sparse_categorical_accuracy.update_state(y_true=labels, y_pred=y_pred)

    print("test accuracy:", sparse_categorical_accuracy.result().numpy())

