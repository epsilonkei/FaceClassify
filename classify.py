import argparse
import numpy as np
from chainer import cuda, serializers
import cv2
from detectFaces import getFaces
from models.CNN import CNN
from train_classify import num_cate, ClasResult

model = CNN(n_class=num_cate)
serializers.load_hdf5('TrainLog/20180803_110721_final/FaceClasModel/FaceCl_050.model', model)


def parser_args():
    parser = argparse.ArgumentParser(description='Face Classify from image')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--image', '-i', default='images/capture.png', help='Image path')
    args = parser.parse_args()
    return args


def classify(args, images, model):
    device_id = args.gpu
    if device_id >= 0:
        xp = cuda.cupy
        model.to_gpu(device_id)
    else:
        xp = np
    X = []
    if len(images) == 0:
        print ('Error: Face not found')
        return
    for im in images:
        X.append(xp.transpose(cv2.resize(im, (64, 64)), (2, 0, 1)))
    X = xp.array(X, dtype=np.float32)
    pred = model.predict(X, train=False)
    if device_id >= 0:
        prediction = ((np.sign(cuda.to_cpu(pred.data)) + 1) / 2).astype(np.int)
    else:
        prediction = ((np.sign(np.array(pred.data)) + 1) / 2).astype(np.int)
    for i, pre in enumerate(prediction):
        result = 'Face {0:2d}: '.format(i)
        for j in range(len(pre)):
            result += ClasResult[j][pre[j]]
            if j < (len(pre) - 1):
                result += ', '
        print (result)


if __name__ == '__main__':
    args = parser_args()
    images = getFaces(cv2.imread(args.image))
    classify(args, images, model)
    for i, img in enumerate(images):
        cv2.imwrite('images/image{0}.jpg'.format(i), img)
