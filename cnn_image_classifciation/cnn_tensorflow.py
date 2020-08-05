import tensorflow as tf
import numpy as np
import os

# load custom imageset directory
data_path_train = r"C:\Users\Administrator\Desktop\datasets\images\flowers\train"
data_path_test = r"C:\Users\Administrator\Desktop\datasets\images\flowers\test"
##tf.enable_eager_execution()


# setup hypervariables for labels and images format
n_classes = 5
img_size = 64
channels = 3

# setup hypervariables for network
learning_rate = 0.0001
epochs = 2
batch_size = 100
drop_rate = 0.6


def _parse_file(data_path):
    imagepaths = []
    labels = []
    label = 0
    classes = sorted(os.walk(data_path).__next__()[1])
    # List each sub-directory (the classes)
    for c in classes:
        c_dir = os.path.join(data_path, c)
        walk = os.walk(c_dir).__next__()

    # Add each image to the training set
        for sample in walk[2]:
            imagepaths.append(os.path.join(c_dir, sample))
            labels.append(label)
        label += 1

    return imagepaths, labels

# unpack inputs for next function
dataset_train = _parse_file(data_path_train)
dataset_test = _parse_file(data_path_test)
test_input = len(dataset_test[0])
train_input = len(dataset_train[0])

# parse and normalize 
def _parse_image(imagepath, label):
    image_string = tf.read_file(imagepath)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [img_size, img_size])
    image = image_resized * (1 / 255)

    return image, label

# create test dataset and iterator
### needs to be streamlined
dataset_test = tf.data.Dataset.from_tensor_slices((dataset_test))
dataset_test = dataset_test.map(_parse_image)
dataset_test = dataset_test.batch(test_input)
test_iterator = dataset_test.make_one_shot_iterator()
test_next = test_iterator.get_next()

# create test dataset and iterator
### needs to be streamlined
dataset = tf.data.Dataset.from_tensor_slices((dataset_train))
dataset = dataset.map(_parse_image)
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(batch_size)
iterator = dataset.make_initializable_iterator()
train_next = iterator.get_next()

##x = tf.placeholder(tf.string, [None])
x = tf.placeholder(tf.float32, [None, img_size, img_size, channels])
y = tf.placeholder(tf.float32, [None, 1])

# hypervariables for layers' output size
K = 8
L = 16
M = 128

# weight, bias with stride size and activation method after convolution for layer 1
W1 = tf.Variable(tf.truncated_normal([6, 6, channels, K], stddev=0.03))
b1 = tf.Variable(tf.truncated_normal([K], stddev=0.01))
stride = 1
y1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1, stride, stride, 1], padding='SAME') + b1)

# weight, bias with stride size and activation method after convolution for layer 2
W2 = tf.Variable(tf.truncated_normal([4, 4, K, L], stddev=0.03))
b2 = tf.Variable(tf.truncated_normal([L], stddev=0.01))
stride = 2  # output is 14x14
y2 = tf.nn.relu(tf.nn.conv2d(y1, W2, strides=[1, stride, stride, 1], padding='SAME') + b2)

yflat = tf.reshape(y2, [-1, 32 * 32 * L])

W3 = tf.Variable(tf.truncated_normal([32 * 32 * L, M], stddev=0.1))
b3 = tf.Variable(tf.truncated_normal([M], stddev=0.01))
y3 = tf.nn.relu(tf.matmul(yflat, W3) + b3)

W4 = tf.Variable(tf.truncated_normal([M, 5], stddev=0.1))
b4 = tf.Variable(tf.truncated_normal([5], stddev=0.01))
ylogits = tf.matmul(y3, W4) + b4
y_ = tf.nn.softmax(ylogits)

# add cross entropy for back prop
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=ylogits, labels=y))

# add an optimiser for back prop
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(r"C:\Users\Administrator\Desktop\pytong\proj\tf_proj")

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    # test batch
    test_x, test_y = sess.run(test_next)
    total_batch = int(train_input / batch_size) + 1
    # define the iterator for the network
    for epoch in range(epochs):
        sess.run(iterator.initializer)
        # holder for the cost
        avg_cost = 0
        for batch in range(batch_size):   
            try:
                batch_x, batch_y = sess.run(train_next)
                # check for correct format and corresponding labels
                print(batch_x)
                print(batch_y)
                op, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: np.expand_dims(batch_y, axis=-1)})
                avg_cost += c / total_batch
                # check if cost is counted properly
                print(avg_cost)

            except tf.errors.OutOfRangeError:
                
                pass
        # verification logic
        test_acc = sess.run(accuracy,feed_dict={x: test_x, y: np.expand_dims(test_y, axis=-1)})
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), " test accuracy: {:.3f}".format(test_acc))
        summary = sess.run(merged, feed_dict={x: test_x, y: np.expand_dims(test_y, axis=-1)})
        writer.add_summary(summary, epoch)

    print("\nTraining complete!")
    writer.add_graph(sess.graph)
    print(sess.run(accuracy, feed_dict={x: test_x, y: np.expand_dims(test_y, axis=-1)}))

