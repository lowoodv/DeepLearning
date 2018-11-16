
import tensorflow as tf
import os

#%% Reading data

def read_data(data_dir, batch_size, shuffle,n_classes,IMG_W_H_D):
    """Read data
    
    Args:
        data_dir: the directory of dataset
        is_train: boolen
        batch_size:
        shuffle:       
    Returns:
        label: 1D tensor, tf.int32
        image: 4D tensor, [batch_size, height, width, 3], tf.float32
    
    """
    img_width = IMG_W_H_D[0]
    img_height = IMG_W_H_D[1]
    img_depth = IMG_W_H_D[2]
    
    
    with tf.name_scope('input'):
        filenames = [os.path.join(data_dir, file_name)
                                        for file_name in os.listdir(data_dir)]          
        filename_queue = tf.train.string_input_producer(filenames)
    
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        img_features = tf.parse_single_example(
                                        serialized_example,
                                        features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                               })
           
        image_raw = tf.decode_raw(img_features['img_raw'], tf.uint8)
        
        label = tf.cast(img_features['label'], tf.int32)
        image = tf.reshape(image_raw, [img_height, img_width,img_depth])         
        image = tf.cast(image, tf.float32)

     
        # data argumentation
#        if shuffle==True:
#            image = tf.image.random_flip_left_right(image)
#            image = tf.image.random_flip_up_down(image)
#            image = tf.image.random_brightness(image, max_delta=63)
#            image = tf.image.random_contrast(image,lower=0.2,upper=1.8)        
        image = tf.image.per_image_standardization(image) #substract off the mean and divide by the variance 


        if shuffle:
            images, label_batch = tf.train.shuffle_batch(
                                    [image, label], 
                                    batch_size = batch_size,
                                    num_threads= 16,
                                    capacity = 20000,
                                    min_after_dequeue = 3000)
        else:
            images, label_batch = tf.train.batch(
                                    [image, label],
                                    batch_size = batch_size,
                                    num_threads = 16,
                                    capacity = 2000)
        ## ONE-HOT      
#        n_classes = 62
        label_batch = tf.one_hot(label_batch, depth= n_classes)
        label_batch = tf.cast(label_batch, dtype=tf.int32)
        label_batch = tf.reshape(label_batch, [batch_size, n_classes])
        
        return images, label_batch
#%%






