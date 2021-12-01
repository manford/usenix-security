import os
import matplotlib as plt
import tensorflow as tf
tf.enable_eager_execution()
print("Tensorflow version is: {}".format(tf.__version__))
print("Eager execution: {}".format((tf.executing_eagerly)))

train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local cache copy of dataset file is: {}".format(train_dataset_fp))

# column name in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

feature_names = column_names[:-1]
label_name = column_names[-1]

print("feature names: {}".format(feature_names))
print("label name: {}".format((label_name)))

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']

batch_size = 32

train_dataset = tf.data.experimental.make_csv_dataset(
    file_pattern=train_dataset_fp,
    batch_size=batch_size, column_names=column_names, label_name=label_name,
    num_epochs=1
)

features, labels = next(iter(train_dataset))

print("features: {}".format(features))
print("labels: {}".format((labels)))

def pack_feature_vector(features, labels):
    """pack feature into single array"""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

train_dataset = train_dataset.map(pack_feature_vector)
features, labels = next(iter(train_dataset))
print("pack features: {}".format(features))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4, )),
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])

print("model variables: {}".format(model.trainable_variables))
logits = model(features)
pre_prob = tf.nn.softmax(logits, axis=1)
pre_class = tf.argmax(logits, axis=1)

print("predict softmax prob: {}".format(pre_prob))
print("labels: {}".format(labels))

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y, training):
    y_ = model(x, training=training)
    return loss_object(y_true=y, y_pred=y_)

l = loss(model, features, labels, training=False)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)  # (y, x)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(), loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {}, Loss: {}".format(optimizer.iterations.numpy(), loss(model, features, labels, training=True)))


train_loss_results = []
train_accuracy_results = []
num_epochs = 21

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy()

    for x, y in train_dataset:
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss_avg.update_state(loss_value)
        epoch_accuracy.update_state(y, model(x, training=True))

        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 5 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))

