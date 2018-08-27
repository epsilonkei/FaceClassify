import os
import argparse
import cv2
from multiprocessing import Pool
from detectFaces import getFaces
from detectAndDraw import load_dlib_predictor
import time

celeba_folder = 'CelebA'
dataset_dir = celeba_folder + '/img_align_celeba'
target_dir = celeba_folder + '/img_align_cropped_celeba'


def parser_args():
    parser = argparse.ArgumentParser(description='Crop face images from images')
    parser.add_argument('--dataset_dir', '-d', default=dataset_dir, help='Dataset directory')
    parser.add_argument('--target_dir', '-t', default=target_dir, help='Target directory')
    parser.add_argument('--process', '-p', default=10, type=int,
                        help='Number of Threads using for Parallel Computing')
    args = parser.parse_args()
    return args


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)


def cropImage(detector, predictor, i, file, args):
    name, ext = os.path.splitext(file)
    if ext in ['.jpg', '.jpeg', '.png', '.gif']:
        face_img = getFaces(detector, predictor, os.path.join(args.dataset_dir, file))
        if len(face_img) > 0:
            target_file = name + time.strftime("%Y%m%d-%H%M%S") + ext
            cv2.imwrite(os.path.join(args.target_dir, target_file), face_img[0])
            #
    if (i % 500 == 0):
        print ('{0: 7d} - {1: 7d}: Start Cropping'.format(i, i + 500))


def wrapperCropImage(argus):
    return cropImage(*argus)


if __name__ == '__main__':
    args = parser_args()
    files = os.listdir(args.dataset_dir)
    createFolder(args.target_dir)
    detector, predictor = load_dlib_predictor()
    _pool = Pool(args.process)
    _pool.map_async(wrapperCropImage, [[detector, predictor, i, file, args] for i, file in enumerate(files)]).get(9999999)  # For Ctrl-C
    _pool.close()
