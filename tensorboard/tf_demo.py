import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../data/',one_hot=True)

#Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
logs_path = '../logs'

#Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256
n_input = 784 #28*28
n_classes = 10 # classes


#tf Graph Input
#mnist data image of shape 28*28
x = tf.placeholder(tf.float32,[None,784],name='InputData')
y = tf.placeholder(tf.float32,[None,10],name='LabelData')

#  Create model
def multilayer_perceptron(x,weights,biases):
    #Hidden layer width RELU activation
    layer_1 = tf.add(tf.matmul(x,weights['w1']),biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    #Create a summary to visualize the first layer ReLU activation
    tf.summary.histogram('relu1',layer_1)
    #hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1,weights['w2']),biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    #Create another summary to visualize the second layer Relu activation
    tf.summary.histogram('relu2',layer_2)
    #Output layer
    out_layer = tf.add(tf.matmul(layer_2,weights['w3']),biases['b3'])
    return out_layer


#Store layers weight & bias
weights = {
    'w1':tf.Variable(tf.random_normal([n_input,n_hidden_1]),name='W1'),
    'w2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2]),name='W2'),
    'w3':tf.Variable(tf.random_normal([n_hidden_2,n_classes]),name='W3')
}

biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden_1]),name='b1'),
    'b2':tf.Variable(tf.random_normal([n_hidden_2]),name='b2'),
    'b3':tf.Variable(tf.random_normal([n_classes]),name='b3')
}


#Encapsulating all ops into scopes,making Tensorboard's Graph
#Visualization more convenient
with tf.name_scope('Model'):
    #Build model
    pred = multilayer_perceptron(x,weights,biases)


with tf.name_scope('Loss'):
    #Softmax Cross entropy(cost funtion)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))

with tf.name_scope('SGD'):
    #Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #Op to calculate every variable gradient
    grads = tf.gradients(loss,tf.trainable_variables())
    grads = list(zip(grads,tf.trainable_variables()))
    #Op to update all variable according to their gradient
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)


with tf.name_scope('Accuracy'):
    #Accuracy
    acc = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    acc = tf.reduce_mean(tf.cast(acc,tf.float32))


#Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
#Create a summary to monitor cost tensor
tf.summary.scalar('loss',loss)
#Create a summary to monitor accuracy tensor
tf.summary.scalar('accuracy',acc)
#Create summaries to visualize weights
for var in tf.trainable_variables():
    tf.summary.histogram(var.name,var)

#Summarize all gradients
for grad,var in grads:
    tf.summary.histogram(var.name + '/gradient',grad)
#Merge all summaries inot a single op
merged_summary_op = tf.summary.merge_all()

#Start training
with tf.Session() as sess:
    #Run the initializer
    sess.run(init)
    #Op to write logs to TensorFlow
    summary_writer = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())
    #Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        #Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #Run optimization op (backprop),cost op (to get loss value)
            _,c,summary = sess.run([apply_grads,loss,merged_summary_op],feed_dict={x:batch_xs,y:batch_ys})
            #Write logs at every iteration
            summary_writer.add_summary(summary,epoch*total_batch + i)
            #Compute average loss
            avg_cost += c/ total_batch
        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print('Epoch:','%04d' %(epoch+1),'cost=','{:.9f}'.format(avg_cost))

    print('optimization Finished!')


    #Test model
    #Calculate accuracy
    print('Accuracy:',acc.eval({x:mnist.test.images,y:mnist.test.labels}))
    print('Run the command line :' + '--> tensorboard --logdir = ../log ,open http://0.0.0.0:6006 into your web browser' )
