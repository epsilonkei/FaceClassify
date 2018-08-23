import argparse
import numpy as np
import chainer
from chainer import serializers, cuda, optimizers, Variable
import json
import time
import os
from models.CNN import CNN
from crop_images import createFolder
from wrap_data import data_dir

train_data = data_dir + '/train_data.npy'
test_data = data_dir + '/test_data.npy'
train_name = data_dir + '/train_name.json'
test_name = data_dir + '/test_name.json'
annotation = data_dir + '/list_attr_celeba.txt'
time_label = time.strftime("%Y%m%d_%H%M%S")
SavedModelFolder = 'TrainLog/' + time_label + '/FaceClasModel'
categories = ['Male', 'Eyeglasses', 'Wearing_Hat', 'Young']
num_cate = len(categories)
ClasResult = [['Female', 'Male'],
              ['No eyeglasses', 'Eyeglasses'],
              ['No wearing hat', 'Wearing hat'],
              ['Not young', 'Young']]
# ClasResult = [['FM', 'M'],
#               ['NE', 'E'],
#               ['NH', 'H'],
#               ['NY', 'Y']]
log_txt = 'TrainLog/' + time_label + '/log.txt'
log_dat = 'TrainLog/' + time_label + '/log.dat'


def read_annotation(anno_file, category):
    label = []
    cate_num = []
    with open(anno_file) as anno:
        for i, line in enumerate(anno):
            if i == 1:
                line_ = line.strip('\n').split(' ')
                for cate in category:
                    cate_num.append(line_.index(cate))
            elif i >= 2:
                line_ = line.strip('\n').split(' ')
                line_ = line_[1:]
                line_ = [int((int(x)+1)/2) for x in line_ if x != '']
                label.append(line_)
    return label, cate_num


def augument(orig):  # 4D tensor
    flip = np.random.randint(2, size=len(orig))*2-1
    theta = np.random.uniform(-0.2, 0.2, len(orig))
    scale = np.random.uniform(0.8, 1.1, len(orig)).reshape(-1, 1, 1)
    shift = np.random.uniform(-2, 2, len(orig)*2).reshape(-1, 2)
    xs = (np.arange(4096) % 4096) % 64
    ys = (np.arange(4096) % 4096)/64
    coords = np.c_[xs, ys].transpose() - 31.5
    R = scale*(np.c_[np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)].reshape(len(orig), 2, 2))
    img = np.array([x[:, np.clip(np.dot(r, coords)[1] + 31.5 + s[1], 0, 63).astype('int32'),
                      np.clip(np.dot(r, coords)[0] + 31.5 + s[0], 0, 63).astype('int32')[::f]].reshape(3, 64, 64)
                    for s, f, r, x in zip(shift, flip, R, orig)])
    return img


def get_data(train_data_file, test_data_file, train_name_file, test_name_file):
    X_train = np.load(train_data_file)
    X_test = np.load(test_data_file)
    with open(train_name_file) as train:
        train_name = json.load(train)
    with open(test_name_file) as test:
        test_name = json.load(test)
    # --- Read annotation ---
    label, cate_num = read_annotation(annotation, categories)
    y_train = []
    for i_train in train_name:
        index = int(i_train[0:6]) - 1
        a = [label[index][i] for i in cate_num]
        # a = label[index][20]
        y_train.append(a)
    y_test = []
    for i_test in test_name:
        index = int(i_test[0:6]) - 1
        a = [label[index][i] for i in cate_num]
        # a = label[index][20]
        y_test.append(a)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, y_train, X_test, y_test


def parser_args():
    parser = argparse.ArgumentParser(
        description='Multi-Task Network for Attribute Classification')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--weight_decay', '-w', type=float,
                        default=0.0001, help='Weight decay ratio')
    args = parser.parse_args()
    return args


def trainClassify():
    args = parser_args()
    device_id = args.gpu

    X_train, y_train, X_test, y_test = get_data(train_data, test_data, train_name, test_name)
    model = CNN(n_class=num_cate)
    if device_id >= 0:
        xp = cuda.cupy
        model.to_gpu(device_id)
    else:
        xp = np

    optimizer = optimizers.MomentumSGD(lr=0.005, momentum=0.9)
    # optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    N_train = len(X_train)
    N_test = len(X_test)
    n_epoch = 50
    batchsize = 30

    with open(log_txt, 'w'):
        pass
    with open(log_dat, 'w'):
        pass
    for epoch in range(n_epoch):
        print ('Epoch %d :' % (epoch+1), end=" ")
        data = open(log_dat, "a")
        data.write('%d ' % (epoch+1))
        data.close()

        # Training
        sum_loss = 0
        pred_y = []
        perm = np.random.permutation(N_train)

        for i in range(0, N_train, batchsize):
            # x = Variable(xp.array(augument(X_train[perm[i: i+batchsize]])).astype(xp.float32))
            x = Variable(xp.array(X_train[perm[i: i+batchsize]]).astype(xp.float32))
            t = Variable(xp.array(y_train[perm[i: i+batchsize]]).astype(xp.int32))

            optimizer.update(model, x, t, True)
            if device_id >= 0:
                sum_loss += cuda.to_cpu(model.loss.data) * len(x.data)
                pred_y.extend((np.sign(cuda.to_cpu(model.y.data)) + 1)/2)
            else:
                sum_loss += np.array(model.loss.data) * len(x.data)
                pred_y.extend((np.sign(np.array(model.y.data)) + 1)/2)

        loss = sum_loss / N_train
        accuracy = xp.sum(xp.array(pred_y, dtype=xp.int32) == xp.array(y_train[perm], dtype=xp.int32)) / N_train / num_cate
        print ('\nTrain loss %.3f, accuracy %.4f |' % (loss, accuracy), end=" ")
        log = open(log_txt, "a")
        log.write('Train loss %.3f, accuracy %.4f |' % (loss, accuracy))
        log.close()
        data = open(log_dat, "a")
        data.write('%.3f %.4f ' % (loss, accuracy))
        data.close()

        for i in range(num_cate):
            accu = xp.sum(xp.array(pred_y, dtype=xp.int32)[:, i] == xp.array(y_train[perm][:, i], dtype=xp.int32))/N_train
            print ('Category %d: %.4f |' % (i, accu), end=" ")
            log = open(log_txt, "a")
            log.write('Category %d: %.4f |' % (i, accu))
            log.close()
            data = open(log_dat, "a")
            data.write('%.4f ' % accu)
            data.close()

        # Testing
        sum_loss = 0
        pred_y = []

        for i in range(0, N_test, batchsize):
            # x = Variable(xp.array(augument(X_test[i: i+batchsize])).astype(xp.float32))
            x = Variable(xp.array(X_test[i: i+batchsize]).astype(xp.float32))
            t = Variable(xp.array(y_test[i: i+batchsize]).astype(xp.int32))
            if device_id >= 0:
                with chainer.using_config('train', False):
                    sum_loss += cuda.to_cpu(model(x, t).data) * len(x.data)
                    pred_y.extend((np.sign(cuda.to_cpu(model.y.data)) + 1)/2)
            else:
                with chainer.using_config('train', False):
                    sum_loss += np.array(model(x, t).data) * len(x.data)
                    pred_y.extend((np.sign(np.array(model.y.data)) + 1)/2)

        loss = sum_loss / N_test
        accuracy = xp.sum(xp.array(pred_y, dtype=xp.int32) == xp.array(y_test, dtype=xp.int32)) / N_test / num_cate
        print ('\n Test loss %.3f, accuracy %.4f |' % (loss, accuracy), end=" ")
        log = open(log_txt, "a")
        log.write('\n Test loss %.3f, accuracy %.4f |' % (loss, accuracy))
        log.close()
        data = open(log_dat, "a")
        data.write('%.3f %.4f ' % (loss, accuracy))
        data.close()

        for i in range(num_cate):
            accu = xp.sum(xp.array(pred_y, dtype=xp.float32)[:, i] == xp.array(y_test[:, i], dtype=xp.float32))/N_test
            print ('Category %d: %.4f |' % (i, accu), end=" ")
            log = open(log_txt, "a")
            log.write('Category %d: %.4f |' % (i, accu))
            log.close()
            data = open(log_dat, "a")
            data.write('%.4f ' % accu)
            data.close()

        print ()
        log = open(log_txt, "a")
        log.write('\n')
        log.close()
        data = open(log_dat, "a")
        data.write('\n')
        data.close()

        if (epoch+1) % 5 == 0:
            serializers.save_hdf5(SavedModelFolder + '/FaceCl_{0:03d}.model'.format(epoch+1), model)


if __name__ == '__main__':
    createFolder(SavedModelFolder)
    trainClassify()
