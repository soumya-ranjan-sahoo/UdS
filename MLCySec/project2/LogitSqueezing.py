
# coding: utf-8

# In[1]:


import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# In[2]:


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import ModelCheckpoint

K.set_image_dim_ordering('th')

from keras.optimizers import *
from keras.wrappers.scikit_learn import KerasClassifier


# In[4]:


#random seed for reproducing the same 
seed = 1234
np.random.seed(seed)


# In[5]:


#reading data
(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[6]:


x_train = x_train.astype("float32")
x_test = x_test.astype("float32")


# In[7]:


x_train = x_train/255.0
x_test = x_test/255.0


# In[8]:


#one hot encoding
y_train_onehot = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)


# In[10]:


K.clear_session()


# In[9]:


trainable_weights = []

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None,28,28,1], name = "X")
    y = tf.placeholder(tf.int32, shape = [None], name = "y")
    training = tf.placeholder(tf.bool, name='training')
    
conv1 = tf.layers.conv2d(X, filters=32, kernel_size = (4,4),
                         strides = (1,1), padding='VALID',
                         activation = tf.nn.relu, name="conv1")

trainable_weights.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1/kernel')[0])
trainable_weights.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv1/bias')[0])

with tf.name_scope("pool1"):
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
     
conv2 = tf.layers.conv2d(pool1, filters=16, kernel_size=(4,4),
                         strides=(1,1), padding="VALID",
                         activation=tf.nn.relu, name="conv2")

trainable_weights.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv2/kernel')[0])
trainable_weights.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'conv2/bias')[0])

with tf.name_scope("pool2"):
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    
drop1 = tf.layers.dropout(pool2, 0.3, training=training, name="drop1")

with tf.name_scope("fc1"):
    drop1_flat = tf.reshape(drop1, shape=[-1,4*4*16])
    fc1 = tf.layers.dense(drop1_flat, 128, activation = tf.nn.relu,
                          name = "fc1")
    
trainable_weights.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc1/kernel')[0])
trainable_weights.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc1/bias')[0])
    
drop2 = tf.layers.dropout(fc1, 0.3, training=training, name="drop2")

with tf.name_scope("fc2"):
    fc2 = tf.layers.dense(drop2, 32, activation = tf.nn.relu,
                          name = "fc2")
    
trainable_weights.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc2/kernel')[0])
trainable_weights.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'fc2/bias')[0])

with tf.name_scope("output"):
    logits = tf.layers.dense(fc2, 10, name = "logits")
    y_proba = tf.nn.softmax(logits, name="y_proba")
    
trainable_weights.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'logits/kernel')[0])
trainable_weights.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'logits/bias')[0])
    
with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


# In[10]:


sess = K.get_session()


# In[13]:


n_epochs = 20
batch_size = 100

with sess.as_default():
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(len(x_train) // batch_size):
            #this cycle is for dividing step by step the heavy work of each neuron
            X_batch = x_train[iteration*batch_size:iteration*batch_size+batch_size,:,:,None]
            y_batch = y_train[iteration*batch_size:iteration*batch_size+batch_size]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch, training: True})
        acc_test = accuracy.eval(feed_dict={X: x_test[:,:,:,None], y: y_test, training: False})
        print("Epoch:",epoch+1, "Train accuracy:", acc_train, "test accuracy:", acc_test)

#         save_path = saver.save(sess, "./my_fashion_model")


# In[14]:


from keras.models import load_model
model = load_model('CNN_model.h5')


# In[15]:


model.evaluate(x_test[:,None,:,:],y_test_onehot,verbose = 1)


# In[18]:


def copy_weights(sess, src_weights, dst_weights):
    with sess.as_default():
        for i in range(len(src_weights)):
            assign_op = dst_weights[i].assign(src_weights[i].eval())
            sess.run(assign_op)


# In[19]:


copy_weights(sess, model.trainable_weights, trainable_weights)


# In[22]:


accuracy.eval(feed_dict={X: x_test[:,:,:,None], y: y_test, training: False}, session=sess)


# In[23]:


with tf.name_scope("train_squeeze"):
    squeeze_term = tf.reduce_mean(logits**2)
    squeeze_loss = tf.reduce_mean(xentropy) +squeeze_term #+ tf.reduce_mean(tf.nn.l2_loss(logits))
    squeeze_optimizer = tf.train.AdamOptimizer()
    squeeze_training_op = squeeze_optimizer.minimize(squeeze_loss)


# In[24]:


with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


# In[25]:


n_epochs = 10
batch_size = 100

with sess.as_default():
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(len(x_train) // batch_size):
            #this cycle is for dividing step by step the heavy work of each neuron
            X_batch = x_train[iteration*batch_size:iteration*batch_size+batch_size,:,:,None]
            y_batch = y_train[iteration*batch_size:iteration*batch_size+batch_size]
            sess.run(squeeze_training_op, feed_dict={X: X_batch, y: y_batch, training: True})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch, training: True})
        acc_test = accuracy.eval(feed_dict={X: x_test[:,:,:,None], y: y_test, training: False})
        print("Epoch:",epoch+1, "Train accuracy:", acc_train, "test accuracy:", acc_test)
       
#         save_path = saver.save(sess, "./my_fashion_model")


# In[26]:


copy_weights(sess, trainable_weights, model.trainable_weights)


# In[34]:


from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import ElasticNetMethod
from cleverhans.utils_keras import KerasModelWrapper


# In[30]:


#loading FGSM from Cleverhans and generating adverserial examples

wrap = KerasModelWrapper(model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.}
adv_x = fgsm.generate_np(x_test[:,None,:,:], **fgsm_params)


# In[32]:


#checking the accuracy of the generated adverserial examples 
adv_pred = np.argmax(model.predict(adv_x), axis = 1)
#advpred_ohe = np_utils.to_categorical(adv_pred)
adv_acc =  np.mean(np.equal(adv_pred, y_test))

print("After attack, the accuracy is: {}".format(adv_acc*100))


# In[35]:


wrap = KerasModelWrapper(model)
en = ElasticNetMethod(wrap, sess=sess)
en_params = {"beta":0.01, 
              "decision_rule":'L1',
              "batch_size":1000, 
              "confidence":0, 
              "learning_rate":0.1, 
              "binary_search_steps":9,
              "max_iterations":10,
              "abort_early":True, 
              "initial_const":0.01, 
              "clip_min":0, 
              "clip_max":1}
adv_x = en.generate_np(x_test[:,None,:,:], **en_params)


# In[37]:


#checking the accuracy of the generated adverserial examples 
adv_conf = model.predict(adv_x)
adv_pred = np.argmax(adv_conf, axis = 1)
adv_acc =  np.mean(np.equal(adv_pred,y_test))


print("After attack, the accuracy is: {}".format(adv_acc*100))

