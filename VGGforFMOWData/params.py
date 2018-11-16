# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 07:36:47 2017

@author: ylu56
"""

load_exsistingmodel=False
IMG_W_H_D =[224,224,3]    #image W/H/D
N_CLASSES = 62              #number of classes
BATCH_SIZE = 32            # batch size
MAX_STEP = 5638000              # Maximum trainiong steps
#Init_Dataset='D:/DATA_fmow/' #path of raw dataset(cheange to your own path)
class_name=['airport','airport_hangar','airport_terminal','amusement_park','aquaculture','archaeological_site','barn','border_checkpoint','burial_site','car_dealership','construction_site','crop_field','dam','debris_or_rubble','educational_institution','electric_substation','factory_or_powerplant','fire_station','flooded_road','fountain','gas_station','golf_course','ground_transportation_station','helipad','hospital','impoverished_settlement','interchange','lake_or_pond','lighthouse','military_facility','multi-unit_residential','nuclear_powerplant','office_building','oil_or_gas_facility','park','parking_lot_or_garage','place_of_worship','police_station','port','prison','race_track','railway_bridge','recreational_facility','road_bridge','runway','shipyard','shopping_mall','single-unit_residential','smokestack','solar_farm','space_facility','stadium','storage_tank','surface_mine','swimming_pool','toll_booth','tower','tunnel_opening','waste_disposal','water_treatment_facility','wind_farm','zoo']
ready_datasetpath='../'    #path of preprocessed dataset(cheange to your own path)
train_datapath=ready_datasetpath+'train' #path of train dataset
val_datapath=ready_datasetpath+'val'     #path of validation dataset
test_datapath=ready_datasetpath+'test'   #test of validation dataset
train_log_dir = ready_datasetpath+'logs/train/'   #training log(cheange to your own path)
val_log_dir = ready_datasetpath+'logs/val/'       #validation log for tensor board(cheange to your own path)
learning_rate = 0.001                        #initial learning rate
Traing_dataset_size=240000                    #Traing_dataset_size
step_per_epoch=Traing_dataset_size/BATCH_SIZE #step_per_epoch
test_dataset_size=84000