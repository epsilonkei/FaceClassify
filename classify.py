import argparse
import numpy as np
from chainer import cuda, serializers
import cv2
import detectFaces
from models.CNN import CNN
from train_classify import num_cate, ClasResult
from utils import visualization
import time

model = CNN(n_class=num_cate)
# serializers.load_hdf5('TrainLog/20180803_110721_final/FaceClasModel/FaceCl_050.model', model)
serializers.load_hdf5('TrainLog/20180813_191926/FaceClasModel/FaceCl_050.model', model)


def parser_args():
    parser = argparse.ArgumentParser(description='Face Classify from image')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--image', '-i', default='images/capture.png', help='Image path')
    parser.add_argument('--out_file', '-o', default='images/image_result.jpg',
                        help='Output image directory')
    args = parser.parse_args()
    return args


def classify(gpu_id, images):
    if gpu_id >= 0:
        xp = cuda.cupy
        model.to_gpu(gpu_id)
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
    if gpu_id >= 0:
        prediction = ((np.sign(cuda.to_cpu(pred.data)) + 1) / 2).astype(np.int)
    else:
        prediction = ((np.sign(np.array(pred.data)) + 1) / 2).astype(np.int)
    for i, pre in enumerate(prediction):
        print ('Face {0:2d}: '.format(i) +
               ', '.join(ClasResult[j][pre[j]] for j in range(len(pre))))


def classifyWithImgResult(gpu_id, org_img, images, bboxes, putText=False):
    if gpu_id >= 0:
        xp = cuda.cupy
        model.to_gpu(gpu_id)
    else:
        xp = np
    X = []
    if len(images) == 0:
        if putText:
            print ('Error: Face not found')
        return org_img
    for im in images:
        X.append(xp.transpose(cv2.resize(im, (64, 64)), (2, 0, 1)))
    X = xp.array(X, dtype=np.float32)
    pred = model.predict(X, train=False)
    if gpu_id >= 0:
        prediction = ((np.sign(cuda.to_cpu(pred.data)) + 1) / 2).astype(np.int)
    else:
        prediction = ((np.sign(np.array(pred.data)) + 1) / 2).astype(np.int)
    captions = ['Face{0:2d}: '.format(i) +
                '\n '.join(ClasResult[j][pre[j]] for j in range(len(pre)))
                for i, pre in enumerate(prediction)]
    if putText:
        for caption in captions:
            print(caption)
    ret = visualization.draw_instance_bboxes(
        img=org_img,
        bboxes=bboxes,
        captions=captions,
    )
    return ret


if __name__ == '__main__':
    args = parser_args()
    start_time = time.process_time()
    # images = detectFaces.getFaces(args.image)
    # classify(args.gpu, images)
    # for i, img in enumerate(images):
    #     cv2.imwrite('images/image{0}.jpg'.format(i), img)
    image = cv2.imread(args.image)
    # face_cascade = detectFaces.load_cascade()
    # f_imgs, bboxes = detectFaces.getFacesWithBorderUsingHaar(face_cascade, image)
    detector, predictor = detectFaces.load_dlib_predictor()
    f_imgs, bboxes = detectFaces.getFacesWithBorderUsingDlib(detector, predictor, image)
    ret_img = classifyWithImgResult(args.gpu, image, f_imgs, bboxes, putText=True)
    cv2.imwrite(args.out_file, ret_img)
    elap_time = time.process_time() - start_time
    print ('Elapsed time: {0:3.2f} seconds'.format(elap_time))
