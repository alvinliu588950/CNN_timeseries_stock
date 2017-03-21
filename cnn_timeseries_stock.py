import tensorflow as tf
import numpy as np
import pandas as pd


# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 40
display_step = 10

#use 5 days data to predict next day price rise or not
#take 5 days and stride 1 day(ex: take 5/1-5/5,then 5/2-5/6)
def strideDataRegenerate(data):
    new_Data = []
    for x in range(0,data.shape[0]-4):
        new_Data = np.append(new_Data,data[x:x+5])

    new_Data = np.reshape(new_Data,(-1,5))
    return new_Data


#shape stock data
def dataProcessing(stockNo):
    data = pd.read_csv('./priceData.csv')
    feed_data = data[['StockNo','Date','Open','High','Low','Close','RiseOrNot']].copy()
    feed_data = feed_data[feed_data['StockNo'] == stockNo]


    #use high,low,close price as 3 channel image
    High_channel = strideDataRegenerate(feed_data.transpose().loc[['High']].transpose().as_matrix()[:674])
    Low_channel = strideDataRegenerate(feed_data.transpose().loc[['Low']].transpose().as_matrix()[:674])
    Close_channel = strideDataRegenerate(feed_data.transpose().loc[['Close']].transpose().as_matrix()[:674])

    #combine 3 channel
    stock_img = []
    for x in range(0,High_channel.shape[0]):
        new_row = np.concatenate([High_channel[x],Low_channel[x],Close_channel[x]],axis=0)
        stock_img = np.append(stock_img,new_row)


    stock_img = np.reshape(stock_img,(-1,3,1,5))
    stock_img_label = np.reshape(feed_data.transpose().loc[['RiseOrNot']].transpose().as_matrix()[5:675],(-1))
    results = {'stock_img' : stock_img,
               'stock_img_label': stock_img_label}


    return results


stock_2427 = dataProcessing(2427)

#train and test data
X_train = stock_2427['stock_img']
Y_train = stock_2427['stock_img_label']

X_test = stock_2427['stock_img'][:400]
Y_test = stock_2427['stock_img_label'][:400]

N = X_train.shape[0]







# Network Parameters
n_input = 5 # 5 days per img
n_classes = 2 # 1 or 0, rise or not
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.int64, shape=[None])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 1, 3, 16])),
    'wc2': tf.Variable(tf.random_normal([4, 1, 16, 32])),
    'wd1': tf.Variable(tf.random_normal([5*32, 512])),
    'out': tf.Variable(tf.random_normal([512, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bd1': tf.Variable(tf.random_normal([512])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}




# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 5, 1, 3])


    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    # conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    # conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, 5*32])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out



# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1),y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1


    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_ind = np.random.choice(N, batch_size, replace=False)
        batch_x = X_train[batch_ind]
        batch_y = Y_train[batch_ind]

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

        step += 1

    print("Optimization Finished!")

    print("pred", \
          sess.run(tf.argmax(pred,1), feed_dict={x: X_test,
                                    y: Y_test,
                                    keep_prob: 1.}))

    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: X_test,
                                      y: Y_test,
                                      keep_prob: 1.}))




