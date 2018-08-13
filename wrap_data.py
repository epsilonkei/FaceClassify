import argparse
import os
import cv2
import numpy as np
import json
from crop_images import target_dir

data_dir = 'data'
train_number = 150000


def parser_args():
    parser = argparse.ArgumentParser(description='Wrap images data to npy and json file')
    parser.add_argument('--images_dir', '-i', default=target_dir, help='Images directory')
    parser.add_argument('--data_dir', '-d', default=data_dir, help='Data directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    train_data = args.data_dir + '/train_data.npy'
    test_data = args.data_dir + '/test_data.npy'
    train_name = args.data_dir + '/train_name.json'
    test_name = args.data_dir + '/test_name.json'
    files = os.listdir(args.images_dir)
    images = []
    names = []
    for i, file in enumerate(files):
        name, ext = os.path.splitext(file)
        if ext in ['.jpg', '.jpeg', '.png', '.gif']:
            img = cv2.imread(os.path.join(args.images_dir, file))
            images.append(cv2.resize(img, (64, 64)).transpose(2, 0, 1))
            names.append(file)

        if (i % 1000 == 0):
            print (i, 'done')

    images = np.array(images)
    perm = np.random.permutation(len(images))
    np.save(train_data, images[perm[:train_number]])
    np.save(test_data, images[perm[train_number:]])

    with open(train_name, 'w') as train:
        json.dump([names[p] for p in perm[:train_number]], train)
    with open(test_name, 'w') as test:
        json.dump([names[p] for p in perm[train_number:]], test)
