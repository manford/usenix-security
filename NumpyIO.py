from __future__ import division, absolute_import, print_function
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd


class NumpyIO:
    """manipulate the tensorflow train and test dataset"""

    row = 0
    num_classes = 0
    train_row = 0
    test_row = 0

    def __init__(self, xtn="the X train file", xtt="the X test file", ytn="the Y train file", ytt="the Y test file"):
        self.xtn = xtn
        self.xtt = xtt
        self.ytn = ytn
        self.ytt = ytt
        self.train_dataset = []
        self.test_dataset = []

    def load(self):
        """
        load raw dataset
        the dtype returned matches the dtypes feeded
        Returns:
            the total dataset
         Raises:
            IOError: an error occurred accessing the bigtable
        """
        # dtypes: int64 and float64
        train_x = pd.read_pickle(self.xtn)
        test_x = pd.read_pickle(self.xtt)
        # dtype: int64
        train_y = np.loadtxt(self.ytn, dtype=int)
        test_y = np.loadtxt(self.ytt, dtype=int)
        # dtype: float64
        train_d = np.hstack((train_x.to_numpy(), train_y.reshape(-1, 1)))
        test_d = np.hstack((test_x.to_numpy(), test_y.reshape(-1, 1)))
        NumpyIO.train_row = len(train_d)
        NumpyIO.test_row = len(test_d)
        return np.vstack((train_d, test_d))

    def __label(self, dataset):
        """encode the label
        Returns:
            LabelEncoder
        """
        label_enc = LabelEncoder()
        label = np.array(label_enc.fit_transform(dataset[:, -1])).reshape(-1, 1)
        print("the labeled labels are: ", label)
        return label

    def getDataset(self):
        dataset = self.load()
        label = self.__label(dataset)
        NumpyIO.num_classes = max(np.ravel(label, 'F')).item() + 1
        # convert numpy.int64 to python int
        print("number of classes is: ", NumpyIO.num_classes)
        train = dataset[:, :-1]  # X part
        dataset = np.hstack((train, label))
        self.train_dataset, self.test_dataset = np.split(dataset, [NumpyIO.train_row])
        return self.train_dataset, self.test_dataset
