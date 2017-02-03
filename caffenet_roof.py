import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

import sys
import os
caffe_root = '/Users/lishiyu/Desktop/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe


if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")

caffe.set_mode_cpu()

model_def = caffe_root + 'examples/roof_recog/models/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)  # use test mode (e.g., don't perform dropout)

blob = caffe.proto.caffe_pb2.BlobProto()
data = open(caffe_root + 'examples/roof_recog/Image_mean/image_mean_train.binaryproto', 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))

mu = arr[0]
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(1, 3, 227, 227)

# image = caffe.io.load_image('roof_images/-2355684.jpg')
# transformed_image = transformer.preprocess('data', image)
# # copy the image data into the memory allocated for the net
# net.blobs['data'].data[...] = transformed_image
# # perform classification
# output = net.forward()
# features = net.blobs['fc7'].data

labels = pd.read_csv('id_train.csv', dtype=object)
train_x = []
for k, filename in enumerate(labels.Id):
    if (k + 1) % 500 == 0:
        print k + 1
    image = caffe.io.load_image('roof_images/' + filename + '.jpg')
    transformed_image = transformer.preprocess('data', image)

    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

    # perform classification
    output = net.forward()

    features = net.blobs['fc7'].data
    train_x.append(features[0].copy())

train_x = np.asarray(train_x)

np.savetxt('caffenet_fc7_feature_batch_1.txt', train_x)


test_y = pd.read_csv('sample_submission4.csv', dtype=object)
test_x = []
for k, filename in enumerate(test_y.Id):
    if (k + 1) % 500 == 0:
        print k + 1
    image = caffe.io.load_image('roof_images/' + filename + '.jpg')
    transformed_image = transformer.preprocess('data', image)

    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

    # perform classification
    output = net.forward()

    features = net.blobs['fc7'].data
    test_x.append(features[0].copy())

test_x = np.asarray(test_x)

np.savetxt('caffenet_fc7_feature_batch_1_test.txt', test_x)



