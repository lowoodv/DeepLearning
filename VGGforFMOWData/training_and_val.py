

#%%
#DATA:
#fmow dataset from IAPRA
    
# TO Train and test:
    #0. get data ready, get paths ready !!!
    #1.run: python run_project.py -train
    #2. un: python run_project.py -val in the console to test
    
#%%

import os
import os.path
import numpy as np
import tensorflow as tf
import input_data
import VGG
import tools

#%%
#IMG_W = 224
#IMG_H = 224
#N_CLASSES = 62
#BATCH_SIZE = 32
#learning_rate = 0.001
#MAX_STEP = 5638000
#global_step=0


#%%   Training
def train_per_epoch(params=None,load_flag=False):
    tf.reset_default_graph()
    IMG_W_H_D=params.IMG_W_H_D
    BATCH_SIZE=params.BATCH_SIZE
    N_CLASSES=params.N_CLASSES
    learning_rate=params.learning_rate
    MAX_STEP=params.MAX_STEP
    global_step=0
    IS_PRETRAIN = True
    train_data_dir = params.train_datapath
    train_log_dir = params.train_log_dir
    epoch_setp=5*params.step_per_epoch
    with tf.name_scope('input'):
        tra_image_batch, tra_label_batch = input_data.read_data(data_dir=train_data_dir,
                                                 batch_size= BATCH_SIZE,
                                                 shuffle=True,n_classes=N_CLASSES,IMG_W_H_D=IMG_W_H_D)        
    logits = VGG.VGG16N(tra_image_batch, N_CLASSES, IS_PRETRAIN)
    loss = tools.loss(logits, tra_label_batch)
    accuracy = tools.accuracy(logits, tra_label_batch)
    
    my_global_step = tf.Variable(0, name='global_step', trainable=False) 
    train_op = tools.optimize(loss, learning_rate, my_global_step)   
    
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    if load_flag==False:
        init = tf.global_variables_initializer()
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True 
    sess = tf.Session(config=config)
    if load_flag:
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(train_log_dir)  ##load checkpoint keep on training
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found. Start from Initial')
            init = tf.global_variables_initializer()
            sess.run(init) 
    else:
        sess.run(init)   
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    
    try:
        for step in np.arange(int(global_step)+1,MAX_STEP):
            if coord.should_stop():
                    break
            _, tra_loss, tra_acc,summary_str= sess.run([train_op, loss, accuracy,summary_op])            
            if step % 500 == 0 or (step + 1) == MAX_STEP:                 
                print ('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                tra_summary_writer.add_summary(summary_str, step)
            if step % epoch_setp == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)   
                break
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
       
    coord.join(threads)
    sess.close()
    return global_step




    
#%%   Test the f1 score on test or validation dataset.
def evaluate(params,dataset='test'):
    tf.reset_default_graph()
    N_classes=params.N_CLASSES
    BATCH_SIZE=params.BATCH_SIZE
    class_name=params.class_name
    confusion_total=np.array([[0 for i in range(N_classes)] for j in range(N_classes)])
    log_dir = params.train_log_dir
    if dataset=='test':
       test_dir = params.test_datapath
    else:
       test_dir = params.val_datapath
    n_test = params.test_dataset_size #num of test samples
    IS_PRETRAIN = False      
    images, labels = input_data.read_data(data_dir=test_dir,
                                              batch_size= BATCH_SIZE,
                                              shuffle=False,n_classes=N_classes,IMG_W_H_D=params.IMG_W_H_D)

    logits = VGG.VGG16N(images, N_classes, IS_PRETRAIN)
    predict=tf.argmax(logits, 1)
    label=tf.argmax(labels, 1)
    c_matrix=tf.confusion_matrix(predict,label,N_classes)
#        correct = tools.num_correct_prediction(logits, labels)
    saver = tf.train.Saver(tf.global_variables())
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True 
    sess = tf.Session(config=config)
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(log_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')
        return
        
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            
    try:
        print('\nEvaluating......')
        num_step = int(n_test/BATCH_SIZE) 
        num_sample = num_step*BATCH_SIZE
        step = 0
        while step < num_step and not coord.should_stop():
            confusion_batch=sess.run(c_matrix) #Batch confusionmatirx
            confusion_total+=confusion_batch                         #Whole confusionmatirx
            step += 1
        print('Total testing samples: %d' %num_sample)
#               print(sess.run(correct))                 
    except Exception as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads)
    F1_total=0
    for i in range(N_classes):
        recall=confusion_total[i,i]/sum(confusion_total[i])
        precision=confusion_total[i,i]/sum(confusion_total[:,i])
        F1_score=2*recall*precision/(recall+precision)
        F1_total+=F1_score
        print('F1 score of class '+class_name[i]+' is: %.3f' %F1_score)
    print("F1 Average %.3f" %(F1_total/N_classes))
        
#%%
def train_val(params=None):
    MAX_STEP=params.MAX_STEP
    load_flag=params.load_exsistingmodel
    global_step=train_per_epoch(params,load_flag)
    evaluate(params,dataset='val')
    while global_step<=MAX_STEP:
        load_flag=True
        global_step=train_per_epoch(params,load_flag)
        evaluate(params,dataset='val')
    print("Reach Max Step!")


