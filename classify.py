import argparse
import numpy as np
import chainer
from chainer import cuda, Variable, Chain, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import cv2
from detectFaces import getFaces

categories = ['Male', 'Eyeglasses', 'Wearing_Hat', 'Young']
num_cate = len(categories)
ClasResult = [['Female', 'Male'],
              ['No eyeglasses', 'Eyeglasses'],
              ['No wearing hat', 'Wearing hat'],
              ['Not young', 'Young']]


class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(3, 32, 3, pad=1),
            conv2=L.Convolution2D(32, 32, 3, pad=1),
            conv3=L.Convolution2D(32, 64, 3, pad=1),
            conv4=L.Convolution2D(64, 64, 3, pad=1),
            conv5=L.Convolution2D(64, 64, 3, pad=1),
            conv6=L.Convolution2D(64, 64, 3, pad=1),
            norm1=L.BatchNormalization(32),
            norm2=L.BatchNormalization(32),
            norm3=L.BatchNormalization(64),
            norm4=L.BatchNormalization(64),
            norm5=L.BatchNormalization(64),
            norm6=L.BatchNormalization(64),
            fc4=L.Linear(4096, 256),
            fc5=L.Linear(256, num_cate)
        )

    def __call__(self, x, t, train):
        h = F.relu(self.norm1(self.conv1(x), test=not train))
        h = F.relu(self.norm2(self.conv2(h), test=not train))
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(h, 0.25, train=train)
        h = F.relu(self.norm3(self.conv3(h), test=not train))
        h = F.relu(self.norm4(self.conv4(h), test=not train))
        h = F.max_pooling_2d(h, 2)
        h = F.dropout(h, 0.25, train=train)
        h = F.relu(self.norm5(self.conv5(h), test=not train))
        h = F.relu(self.norm6(self.conv6(h), test=not train))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.fc4(h))
        h = F.dropout(h, 0.25, train=train)
        self.y = self.fc5(h)
        self.loss = F.sigmoid_cross_entropy(self.y, t)
        return self.loss, self.y


model = CNN()
# serializers.load_hdf5('FaceClasModel/FaceCl_004.model', model)
# serializers.load_hdf5('TrainLog/20180802_/FaceClasModel/FaceCl_044.model', model)
serializers.load_hdf5('TrainLog/20180802_230648_augument/FaceClasModel/FaceCl_045.model', model)


def classify(images, model):
    X = []
    if len(images) == 0:
        print ('Error: Face not found')
        return
    for im in images:
        X.append(np.transpose(cv2.resize(im, (64, 64)), (2, 0, 1)))
    t = np.zeros((len(X), num_cate))
    X = np.array(X, dtype=np.float32)
    t = np.array(t, dtype=np.int32)
    _, pred = model(X, t, train=False)
    prediction = ((np.sign(np.array(pred.data)) + 1) / 2).astype(np.int)
    for i, pre in enumerate(prediction):
        result = 'Face {0:2d}: '.format(i)
        for j in range(len(pre)):
            result += ClasResult[j][pre[j]]
            if j < (len(pre) - 1):
                result += ', '
        print (result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Classify from image')
    parser.add_argument('--image', '-i', default='images/capture.png', help='Image path')
    args = parser.parse_args()
    images = getFaces(cv2.imread(args.image))
    classify(images, model)
    for i, img in enumerate(images):
        cv2.imwrite('images/image{0}.jpg'.format(i), img)
