# %%
# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# model.save("./model.h5");

# %%
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
# %%
train_images = train_images / 255.0

test_images = test_images / 255.0
# %%
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()
# %%
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
# %%
model.fit(train_images, train_labels, epochs=10)
# %%
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
# %%
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

probability_model.compile()
# model.compile(loss="categorical_crossentropy",
#               optimizer="adam",
#               metrics=["acc"])
probability_model.save("model.h5");
# predictions = probability_model.predict(test_images)

# np.argmax(predictions[0])
# %%
# test_labels[0]
# %%
