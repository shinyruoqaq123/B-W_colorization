import tensorflow as tf
import os
import random
import sys

num_epochs = 4
batch_size = 32
learning_rate = 0.001

data_dir = sys.argv[1]
train_cats_dir = data_dir + 'cats/'
train_dogs_dir = data_dir + 'dogs/'
train_panda_dir=data_dir+'panda/'

def _decode_resize(filename, label):
    image_string = tf.io.read_file(filename)
    image_docoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize(image_docoded, [256, 256]) / 255.0
    return image_resized, label


if __name__ == '__main__':
    train_cat_filenames = [train_cats_dir + filename for filename in os.listdir(train_cats_dir)]
    train_dog_filenames = [train_dogs_dir + filename for filename in os.listdir(train_dogs_dir)]
    train_panda_filenames = [train_panda_dir + filename for filename in os.listdir(train_panda_dir)]
    train_filenames = train_cat_filenames+train_dog_filenames+train_panda_filenames
    random.shuffle(train_filenames)

    # 0:cat, 1:dog, 2:panda
    train_labels = []
    for filename in train_filenames:
        if filename.split('/')[-2]=='cats':
            train_labels.append(0)
        elif filename.split('/')[-2]=='dogs':
            train_labels.append(1)
        else:
            train_labels.append(2)

    train_filenames=tf.constant(train_filenames)
    train_labels=tf.constant(train_labels,dtype=tf.int32)
    print("train_filenames:",train_filenames)
    print("train_labels:",train_labels)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(map_func=_decode_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_dataset=train_dataset.repeat(100)
    train_dataset = train_dataset.shuffle(buffer_size=100)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # build model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, 5, activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_crossentropy,tf.keras.metrics.sparse_categorical_accuracy]
    )

    model.fit(train_dataset, epochs=num_epochs)

    model.save('../../checkpoint/classifier/model.h5')


