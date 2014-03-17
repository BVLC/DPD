#from decaf.scripts.jeffnet import JeffNet
from decaf.scripts import imagenet
from skimage import io
import numpy as np
import scipy.io as sio
import os
import sys

#data_root = '/u/vis/common/deeplearning/models/'
data_root='/home/forrest/decaf/models/'
#net = JeffNet(data_root+'imagenet.jeffnet.epoch90', data_root+'imagenet.jeffnet.meta')
net = imagenet.DecafNet(data_root+'imagenet.jeffnet.epoch90', data_root+'imagenet.jeffnet.meta')
if len(sys.argv)!=4:
    print "Usage ", sys.argv[0], " <image_dir> <feature_path> <num_imgs>"
    #print "Usage ", sys.argv[0], " <image_dir> <feature_path>" #no longer need to input num_imgs ... we sort this out below
    exit(1)
img_dir = sys.argv[1]
feature_path = sys.argv[2]
max_imgs = int(sys.argv[3])
#image_list = os.listdir(img_dir) #assume all the files in img_dir are indeed images
#image_list = sorted(image_list) 
#max_imgs=len(image_list)

features = np.zeros((max_imgs,4096))
for i in range(max_imgs):
    #filename = img_dir + '/' + "%04d.jpg"  %(i+1) #TODO: automatically get list of filenames using os.walk()
    filename = img_dir + '/' + "%05d.jpg"  %(i+1) 
    #filename = img_dir + '/' + image_list[i]
    sys.stdout.write("Extracting DeCAF features from image %s\n" %filename)

    if os.path.exists(filename):
        img = io.imread(filename)
        net.classify(img, center_only=True)
        features[i,:] = net.feature('fc6_cudanet_out')
    else:
        print "WARNING: DeCAF found an missing or invalid image file, %s" % filename

sio.savemat(feature_path,{'features':features})

    

