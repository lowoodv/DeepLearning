Tensorflow implementation of [VGG16](https://arxiv.org/abs/1608.06993) using **FMOW Datset**
* The code that implements *this paper* is ***VGG.py***
## Requirements
* Tensorflow 1.x
* Python 3.x
To run the code 
1.Change the parameters in params.py to local settings on you computer;
2.Before  traning or test
run: python runBaseline.py -prepare_data 
It will datasets train val and test. This process may take several hours.
3.After the dataset is ready
python run_project.py -train (run run_project.py -train# in Ipython console)#can be used to starte training the model and at the same time you can use the command:tensorborad logdir=D:\tfrecord\logs\ in a new command line window under the log dir to start the tensorbord can check the traning process by internet explore(Type in:"http://localhost:6006"). 
python run_project.py -evl	(run run_project.py -evl# in Ipython console) #can be used to generate F1-score per class and average F1-score(This function need a pretrained model in the format of checkpoint in the train_log_dir).
A pretrained 300000 steps model is in ./logs/train in the format of checkpoint
