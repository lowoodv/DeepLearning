# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 07:19:49 2017

@author: ylu56
"""
import training_and_val
import data_preprocess
class project_main:
    def __init__(self, params=None, argv=None):
        self.params = params
        for arg in argv:
            if arg == '-prepare_data':
                data_preprocess.prepare_date(params)
            if arg == '-train':
                training_and_val.train_val(params)
            if arg == '-evl':
                training_and_val.evaluate(params)
                
                