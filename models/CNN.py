from chainer import Chain
import chainer.functions as F
import chainer.links as L


class CNN(Chain):
    def __init__(self, n_class=4):
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
            fc5=L.Linear(256, n_class)
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
        return self.loss

    def predict(self, x, train=False):
        h = F.relu(self.norm1(self.conv1(x), test=not train))
        h = F.relu(self.norm2(self.conv2(h), test=not train))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.norm3(self.conv3(h), test=not train))
        h = F.relu(self.norm4(self.conv4(h), test=not train))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.norm5(self.conv5(h), test=not train))
        h = F.relu(self.norm6(self.conv6(h), test=not train))
        h = F.max_pooling_2d(h, 2)
        h = F.relu(self.fc4(h))
        self.y = self.fc5(h)
        return self.y
