import caffe  
  
import numpy as np  
  


np.set_printoptions(threshold='nan')  
  

MODEL_FILE = '/home/pry/DeepCompression-caffe-master/examples/mnist/lenet_train_test_compress_stage3.prototxt'  

PRETRAIN_FILE = '/home/pry/DeepCompression-caffe-master/examples/mnist/models/lenet_finetune_stage3_iter_4000.caffemodel'  
  

params_txt = 'params.txt'  
pf = open(params_txt, 'w')  
  

net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)  
  

for param_name in net.params.keys():  
  
    weight = net.params[param_name][0].data  
 
    bias = net.params[param_name][1].data  
  
    
    pf.write(param_name)  
    pf.write('\n')  
  
 
    pf.write('\n' + param_name + '_weight:\n\n')  
   
    weight.shape = (-1, 1)  
  
    for w in weight:  
        pf.write('%ff, ' % w)  
  
   
    pf.write('\n\n' + param_name + '_bias:\n\n')  
    
    bias.shape = (-1, 1)  
    for b in bias:  
        pf.write('%ff, ' % b)  
  
    pf.write('\n\n')    
pf.close 
