import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
x_train, x_test = x_train.reshape([-1, 784]), x_test.reshape([-1, 784])
x_train, x_test = x_train / 255., x_test / 255.
y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(60000).batch(50)
train_data_iter = iter(train_data)

class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')
        self.pool_layer_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.conv_layer_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')
        self.pool_layer_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

        self.flatten_layer = tf.keras.layers.Flatten()
        self.fc_layer_1 = tf.keras.layers.Dense(1024, activation='relu')
        self.output_layer = tf.keras.layers.Dense(10, activation=None)

    def call(self, x):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        h_conv1 = self.conv_layer_1(x_image)
        h_pool1 = self.pool_layer_1(h_conv1)
        h_conv2 = self.conv_layer_2(h_pool1)
        h_pool2 = self.pool_layer_2(h_conv2)
        flatten = self.flatten_layer(h_pool2)
        h_fc = self.fc_layer_1(flatten)
        logits = self.output_layer(h_fc)
        y_pred = tf.nn.softmax(logits)

        return logits, y_pred
    
@tf.function
def cross_entropy_loss(logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

optimizer = tf.optimizers.Adam(1e-4)

train_summary_writer = tf.summary.create_file_writer('./tensorboard_log/train')
test_summary_writer = tf.summary.create_file_writer('./tensorboard_log/test')

@tf.function
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        y_pred, logits = model(x)
        loss = cross_entropy_loss(logits, y)
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', loss, step=optimizer.iterations)
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('training image', x_image, max_outputs=10, step=optimizer.iterations)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
def compute_accuracy(y_pred, y, summary_writer):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with summary_writer.as_default():
        tf.summary.scalar('accuracy', accuracy, step=optimizer.iterations)

    return accuracy

CNN_model = CNN()

for i in range(10000):
    batch_x, batch_y = next(train_data_iter)

    if i % 100 == 0:
        train_accuracy = compute_accuracy(CNN_model(batch_x)[0], batch_y, train_summary_writer)
        print("반복(Epoch): %d, 트레이닝 데이터 정확도: %f" % (i+100, train_accuracy))

print("정확도(Accuracy): %f", compute_accuracy(CNN_model(x_test)[0], y_test, test_summary_writer))