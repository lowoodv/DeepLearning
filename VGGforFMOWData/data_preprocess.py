# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:45:22 2017

@author: ylu56
"""
import numpy as np
import os,fnmatch
import tensorflow as tf
from PIL import Image
import json
Image.MAX_IMAGE_PIXELS = 1000000000   
def record_example(class_path,label,img_name):
    """
    Input: class_path,label     
    Output: TFrecord example
    """
    Image.MAX_IMAGE_PIXELS = 1000000000
    img_path = class_path+'/'+img_name
    img = Image.open(img_path)
    jsfile_path=class_path+'/'+os.path.splitext(img_name)[0]+'.json'
    with open(jsfile_path, 'r') as f:
        jsfile = json.load(f)
    area=jsfile['bounding_boxes'][0]['box']
    if area[2]<=2 or area[3]<=2: return
    widthBuffer = int((area[2] * 0.5) / 2.0)
    heightBuffer = int((area[3] * 0.5) / 2.0)
    ca1=area[0]-widthBuffer
    ca2=area[1]-heightBuffer
    ca3=area[0]+area[2]+widthBuffer
    ca4=area[1]+area[3]+heightBuffer
    if ca1<0: ca1=0
    if ca2<0: ca2=0
    if ca3>img.size[0]: ca3=img.size[0]
    if ca4>img.size[1]: ca4=img.size[1]
    crop_area=(ca1,ca2,ca3,ca4)
    img_crop = img.crop(crop_area) 
    img_resize = img_crop.resize((224,224))
    img_raw=img_resize.tobytes()
    example = tf.train.Example(
        features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
    }))
    return example
def tfrecord_convert(dataset_path,tffile_path,dataset_type,num_records_ineach,recent_tffile):
    """
    Convert image and metadata in to tfrecords
    Input: dataset file path tf file path dataset type,num_records_ineach
    Output print convert finished
    """
    data_type=['train','val','test']
    data_classes=os.listdir(dataset_path)
    num_classes=len(data_classes)
    recent_class=0
    totl_num_records=0
    finish_flag=[ 0 for n_flag in np.arange(0, num_classes)]
    second_dir_indicator=[ 0 for n_class in np.arange(0, num_classes)]
    image_name_indicator=[ 0 for n_second_dir in np.arange(0, num_classes)]
    randnum=np.random.randint(0, 5)
    if randnum>=2: 
        data_type_indcator=2
    else:
       data_type_indcator=randnum
    tf_filename=tffile_path+data_type[data_type_indcator]+"/"+dataset_type+str(recent_tffile)+".tfrecords"
    writer = tf.python_io.TFRecordWriter(tf_filename)
    while(1):
        if 0 not in finish_flag:
            break
        if finish_flag[recent_class]==1:
            if recent_class==num_classes-1: recent_class=-1
            recent_class=recent_class+1
            continue
        if totl_num_records>num_records_ineach:
           writer.close()
           totl_num_records=0
           recent_tffile=recent_tffile+1
           randnum=np.random.randint(0, 5)
           if randnum>=3: 
              data_type_indcator=3
           else:
              data_type_indcator=randnum
           tf_filename=tffile_path+data_type[data_type_indcator]+"/"+dataset_type+str(recent_tffile)+".tfrecords"
           writer = tf.python_io.TFRecordWriter(tf_filename)
        class_name=data_classes[recent_class]
        second_dir_list=os.listdir(dataset_path+'/'+class_name)
        if len(second_dir_list)==second_dir_indicator[recent_class]:
           finish_flag[recent_class]=1
           if recent_class==num_classes-1: recent_class=-1
           recent_class=recent_class+1
           continue
        class_path=dataset_path+'/'+class_name+'/'+second_dir_list[second_dir_indicator[recent_class]]
        img_name_list=fnmatch.filter(os.listdir(class_path), '*_rgb.jpg')
        if len(img_name_list)==image_name_indicator[recent_class]:
           image_name_indicator[recent_class]=0
           second_dir_indicator[recent_class]=second_dir_indicator[recent_class]+1
           continue
        img_name=img_name_list[image_name_indicator[recent_class]]
        label=recent_class
        TF_record=record_example(class_path,label,img_name)
        if TF_record==None:
           image_name_indicator[recent_class]=image_name_indicator[recent_class]+1
           continue
        writer.write(TF_record.SerializeToString())
        image_name_indicator[recent_class]=image_name_indicator[recent_class]+1
        if recent_class==num_classes-1: recent_class=-1
        recent_class=recent_class+1
        totl_num_records=totl_num_records+1
    return recent_tffile+1
def prepare_date(params=None):
    dataset_path=params.Init_Dataset
    destination_path=params.ready_datasetpath
    num_records_ineach=6000
    dataunit_name="fowmdataunit"
    num_dataunit=0
    num_dataunit=tfrecord_convert(dataset_path+'train',destination_path,dataunit_name,num_records_ineach,num_dataunit)
    tfrecord_convert(dataset_path+'val',destination_path,dataunit_name,num_records_ineach,num_dataunit)
    print('Data Ready!')
        
        
        
           
        
        
