import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import svm

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os
if not os.path.isfile(caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")
    #!../scripts/download_model_binary.py ../models/bvlc_reference_caffenet

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + 'models/bvlc_alexnet/deploy.prototxt',
                caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
net.blobs['data'].reshape(1,3,227,227)

features_fc7_train = []
features_fc8_train = []

features_fc7_test = []
features_fc8_test = []

features_fc7_fc8_train = []
features_fc7_fc8_test = []


labels = sio.loadmat('necktie_GT.mat')
labels = labels['GT']
labels = labels - 1
labels_train = labels[:1328,0]
labels_test = labels[1328:,0]

file_object = open("output.txt",'w')

#The below code is for extracting features from alexnet network and features are being extracted from fc7,fc8
f = open('train.txt','r') # This is for training the classifier
for line in f:
    line1 = line.split()
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(line1[0]))
    out = net.forward()
    feat = np.array(net.blobs['fc7'].data[0])
    feat1 = np.array(net.blobs['fc8'].data[0]) #Creating a new array each time because feat is the reference of the net.blobs... and
    features_fc7_train.append(feat)           # each time blob changes, the feat also changes. therefore, to avoid this.
    features_fc8_train.append(feat1)
    feat = np.append(feat,feat1)
    features_fc7_fc8_train.append(feat)
features_fc7_train = np.array(features_fc7_train)
features_fc8_train = np.array(features_fc8_train)
features_fc7_fc8_train = np.array(features_fc7_fc8_train)


f1 = open('test.txt','r') #This is for creating a test file

for line in f1:
    line1 = line.split()
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(line1[0]))
    out = net.forward()
    feat = np.array(net.blobs['fc7'].data[0])
    feat1 = np.array(net.blobs['fc8'].data[0]) #Creating a new array each time because feat is the reference of the net.blobs... and
    features_fc7_test.append(feat)           # each time blob changes, the feat also changes. therefore, to avoid this.
    features_fc8_test.append(feat1)
    feat = np.append(feat,feat1)
    features_fc7_fc8_test.append(feat)
features_fc7_test = np.array(features_fc7_test)
features_fc8_test = np.array(features_fc8_test)
features_fc7_fc8_test = np.array(features_fc7_fc8_test)

'''
Now that we have extracted features we will be training a classfier. We will be generating a text file for
that will contain various result
'''
print("This is the file to compare results:\n")
file_object.write("This is the file to compare results:\n")

print("First: Using first 500 features fc7 layer to train svm\n")
file_object.write("First: Using first 500 features fc7 layer to train svm\n")
C=1.0
svc = svm.SVC(kernel='linear', C=C).fit(features_fc7_train[:,:500], labels_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(features_fc7_train[:,:500], labels_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(features_fc7_train[:,:500], labels_train)
lin_svc = svm.LinearSVC(C=C).fit(features_fc7_train[:,:500], labels_train)


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    print("Classifier: ", clf)
    Z = clf.predict(features_fc7_test[:,:500])
    scores = (Z==labels_test)
    summation = np.sum(scores)
    print("\nAccuracy: ",summation/528.0)
    file_object.write("\nAccuracy: %f" % (summation/528.0))


file_object.write("\n\n")

print('\nUsing first 1500 features of fc7 layer to train svm')
file_object.write('\nUsing first 1500 features of fc7 layer to train svm')
C=1.0
svc = svm.SVC(kernel='linear', C=C).fit(features_fc7_train[:,:1500], labels_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(features_fc7_train[:,:1500], labels_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(features_fc7_train[:,:1500], labels_train)
lin_svc = svm.LinearSVC(C=C).fit(features_fc7_train[:,:1500], labels_train)


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    print("\nClassifier: ", clf)
    Z = clf.predict(features_fc7_test[:,:1500])
    scores = (Z==labels_test)
    summation = np.sum(scores)
    print("\nAccuracy: ",summation/528.0)
    file_object.write("\nAccuracy: %f" % (summation/528.0))

file_object.write("\n\n")

print('\nUsing half features of fc7 layer to train svm')
file_object.write('\nUsing half features of fc7 layer to train svm')
C=1.0
svc = svm.SVC(kernel='linear', C=C).fit(features_fc7_train[:,:2048], labels_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(features_fc7_train[:,:2048], labels_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(features_fc7_train[:,:2048], labels_train)
lin_svc = svm.LinearSVC(C=C).fit(features_fc7_train[:,:2048], labels_train)


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    print("\nClassifier: ", clf)
    Z = clf.predict(features_fc7_test[:,:2048])
    scores = (Z==labels_test)
    summation = np.sum(scores)
    print("\nAccuracy: ",summation/528.0)
    file_object.write("\nAccuracy: %f" % (summation/528.0))

file_object.write("\n\n")

print('\nFouth Taking all the features of fc7 layer')
file_object.write('\nFouth Taking all the features of fc7 layer')
C=1.0
svc = svm.SVC(kernel='linear', C=C).fit(features_fc7_train, labels_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(features_fc7_train, labels_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(features_fc7_train, labels_train)
lin_svc = svm.LinearSVC(C=C).fit(features_fc7_train, labels_train)


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    print("\nClassifier: ", clf)
    Z = clf.predict(features_fc7_test)
    scores = (Z==labels_test)
    summation = np.sum(scores)
    print("\nAccuracy",summation/528.0)
    file_object.write("\nAccuracy: %f" % (summation/528.0))

file_object.write("\n\n")

print('\nTaking all the features of fc8 layer')
file_object.write('\nTaking all the features of fc8 layer')
C=1.0
svc = svm.SVC(kernel='linear', C=C).fit(features_fc8_train, labels_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(features_fc8_train, labels_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(features_fc8_train, labels_train)
lin_svc = svm.LinearSVC(C=C).fit(features_fc8_train, labels_train)


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    print("\nClassifier: ", clf)
    Z = clf.predict(features_fc8_test)
    scores = (Z==labels_test)
    summation = np.sum(scores)
    print("\nAccuracy: ",summation/528.0)
    file_object.write("\nAccuracy: %f" % (summation/528.0))

file_object.write("\n\n")

print('\nTaking all the features of fc8 and fc7 layer')
file_object.write('\nTaking all the features of fc8 and fc7 layer')
C=1.0
svc = svm.SVC(kernel='linear', C=C).fit(features_fc7_fc8_train, labels_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(features_fc7_fc8_train, labels_train)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(features_fc7_fc8_train, labels_train)
lin_svc = svm.LinearSVC(C=C).fit(features_fc7_fc8_train, labels_train)


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    print("\nClassifier: ", clf)
    Z = clf.predict(features_fc7_fc8_test)
    scores = (Z==labels_test)
    summation = np.sum(scores)
    print("\nAccuracy: ",summation/528.0)
    file_object.write("\nAccuracy: %f" % (summation/528.0))
