import os
import cv2
import numpy as np
import json

dir = '../fin_celebA'
data_dir = 'data'
train_data = data_dir + '/train_data.npy'
test_data = data_dir + '/test_data.npy'
train_name = data_dir + '/train_name.json'
test_name = data_dir + '/test_name.json'


if __name__ == '__main__':
    files = os.listdir(dir)

    images = []
    names = []
    for i, file in enumerate(files):
        name, ext = os.path.splitext(file)
        if ext in ['.jpg', '.jpeg', '.png', '.gif']:
            img = cv2.imread(os.path.join(dir, file))
            images.append(cv2.resize(img, (64, 64)).transpose(2, 0, 1))
            names.append(file)

        if (i % 1000 == 0):
            print (i, 'done')

    images = np.array(images)
    perm = np.random.permutation(len(images))
    np.save(train_data, images[perm[:80000]])
    np.save(test_data, images[perm[80000:]])

    with open(train_name, 'w') as train:
        json.dump([names[p] for p in perm[:80000]], train)
    with open(test_name, 'w') as test:
        json.dump([names[p] for p in perm[80000:]], test)
