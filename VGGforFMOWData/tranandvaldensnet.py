# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:09:27 2017

@author: ylu56
"""

import os
import os.path

import numpy as np
import tensorflow as tf
import Densenet
import input_data
#import VGG
import tools

#%%
IMG_W = 224
IMG_H = 224
N_CLASSES = 62
BATCH_SIZE = 1
init_learning_rate = 1e-4
MAX_STEP = 5638000   # it took me about one hour to complete the training.
IS_PRETRAIN = False
growth_k = 24
nb_block = 2
epsilon = 1e-4
#%%   Training
def train():
#    m=0
    data_dir = 'D:/tfrecorddata/'
    train_log_dir = 'D:/tfrecorddata/logs/train_densnet/'
#    val_log_dir = 'D:/tfrecorddata/logs/val/'
    
    with tf.name_scope('input'):
        tra_image_batch, tra_label_batch = input_data.read_cifar10(data_dir=data_dir,
                                                 is_train=True,
                                                 batch_size= BATCH_SIZE,
                                                 shuffle=True)
        val_image_batch, val_label_batch = input_data.read_cifar10(data_dir=data_dir,
                                                 is_train=False,
                                                 batch_size= BATCH_SIZE,
                                                 shuffle=False)
        
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, N_CLASSES]) 
    training_flag = tf.placeholder(tf.bool)
    logits = Densenet_Cifar10.DenseNet(x=x, nb_blocks=nb_block, filters=growth_k, training=training_flag).model
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)
    
#    my_global_step = tf.Variable(0, name='global_step', trainable=False) 
    optimizer = tf.train.AdamOptimizer(learning_rate=init_learning_rate, epsilon=epsilon)
    train_op = optimizer.minimize(loss)   
    
    saver = tf.train.Saver(tf.global_variables())
#    summary_op = tf.summary.merge_all()   
       
    init = tf.global_variables_initializer()
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True 
    sess = tf.Session(config=config)
#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333) 
#    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options)) 
    sess.run(init)
    
    # load the parameter file, assign the parameters, skip the specific layers
    # tools.load_with_skip(pre_trained_weights, sess, ['fc6','fc7','fc8'])   


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
#    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
#    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
                
            tra_images,tra_labels = sess.run([tra_image_batch, tra_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x:tra_images, y_:tra_labels,training_flag : True})            
            if step % 5000 == 0 or (step + 1) == MAX_STEP:                 
                print ('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
#                   checkpoint_path = os.path.join(train_log_dir, 'model%d.ckpt' %m)
#                   saver.save(sess, checkpoint_path, global_step=step)
#                   m=m+1
#                summary_str = sess.run(summary_op)
#                tra_summary_writer.add_summary(summary_str, step)
            if step % 11280 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={x:val_images,y_:val_labels,training_flag : True})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc))

#                summary_str = sess.run(summary_op)
#                val_summary_writer.add_summary(summary_str, step)
                    
            if step % 20000  == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()