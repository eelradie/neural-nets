import keras.layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

print(tfds.list_builders())
tfds.disable_progress_bar()
# get info on on the dataset
builder = tfds.builder('rock_paper_scissors')
info = (builder.info)

###### Prepp the dateset #####
df_train, df_test = tfds.load(name='rock_paper_scissors', split=['train', 'test'])

# tfds.show_examples(info, df_train)

train_images = np.array(
    [ds['image'].numpy()[:, :, 0] for ds in df_train])  # to get only one color because we are training on the edges
train_labels = np.array([ds['label'].numpy() for ds in df_train])

test_images = np.array(
    [ds['image'].numpy()[:, :, 0] for ds in df_test])  # to get only one color because we are training on the edges
test_labels = np.array([ds['label'].numpy() for ds in df_test])

# print((train_images[0]), " ", len(train_labels))
# print(train_images.shape)
# print(test_images.shape)

# since we are using gray-scale images we will let keras know that by rehaping that to (# of examples, x-axis pixels, y-axis pixels, 1) where 1 is for how many color channels we are having (convolutional neural networks properties)
train_images = train_images.reshape(2520, 300, 300, 1)
test_images = test_images.reshape(372, 300, 300, 1)

# now we need to change the scale from 0-255 to 0-1
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_images /= 255
test_images /= 255

print(train_labels)

###### train the model #####

model = keras.Sequential([keras.layers.Flatten(),
                          keras.layers.Dense(512, activation=tf.keras.activations.relu),
                          keras.layers.Dense(256, activation=tf.keras.activations.relu),
                          keras.layers.Dense(3, activation=tf.keras.activations.softmax)])

# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#               metrics=tf.keras.metrics.Accuracy())


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=32)

print("Evaluation")
model.evaluate(test_images, test_labels)