import os
import argparse
import cv2
from multiprocessing import Pool
from detectFaces import getFaces
import time

celeba_folder = 'CelebA'
dataset_dir = celeba_folder + '/img_align_celeba'
target_dir = celeba_folder + '/img_align_cropped_celeba'


def parser_args():
    parser = argparse.ArgumentParser(description='Crop face images from images')
    parser.add_argument('--dataset_dir', '-d', default=dataset_dir, help='Dataset directory')
    parser.add_argument('--target_dir', '-t', default=target_dir, help='Target directory')
    args = parser.parse_args()
    return args


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)


def cropImage(i, file, args):
    name, ext = os.path.splitext(file)
    if ext in ['.jpg', '.jpeg', '.png', '.gif']:
        face_img = getFaces(os.path.join(args.dataset_dir, file))
        if len(face_img) > 0:
            target_file = name + time.strftime("%Y%m%d-%H%M%S") + ext
            cv2.imwrite(os.path.join(args.target_dir, target_file), face_img[0])
            #
    if (i % 200 == 0):
        print ('{0: 7d} Done'.format(i))


def wrapperCropImage(argus):
    return cropImage(*argus)


if __name__ == '__main__':
    args = parser_args()
    files = os.listdir(args.dataset_dir)
    createFolder(args.target_dir)
    p = Pool(16)
    p.map(wrapperCropImage, [[i, file, args] for i, file in enumerate(files)])
