import numpy as np
import argparse
from typing import Callable, List, Tuple


parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')


def args2data(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
str, str, str, int, int, int, float]:
    """
    :return:
    X_tr: train data *without label column and without bias folded in
        (numpy array)
    y_tr: train label (numpy array)
    X_te: test data *without label column and without bias folded in*
        (numpy array)
    y_te: test label (numpy array)
    out_tr: file for predicted output for train data (file)
    out_te: file for predicted output for test data (file)
    out_metrics: file for output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """
    # Get data from arguments
    out_tr = args.train_out
    out_te = args.validation_out
    out_metrics = args.metrics_out
    n_epochs = args.num_epoch
    n_hid = args.hidden_units
    init_flag = args.init_flag
    lr = args.learning_rate

    X_tr = np.loadtxt(args.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr = X_tr[:, 1:]  # cut off label column

    X_te = np.loadtxt(args.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te = X_te[:, 1:]  # cut off label column

    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)


def shuffle(X, y, epoch):
    """
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]


def zero_init(shape):
    """
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    return np.zeros(shape=shape)


def random_init(shape):
    """
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    M, D = shape
    np.random.seed(M * D)
    return np.random.uniform(-0.1,0.1,size=shape)


class SoftMaxCrossEntropy:

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        :param z: input logits of shape (num_classes,)
        :return: softmax output of shape (num_classes,)
        """
        expz = np.exp(z)
        return expz/np.sum(expz)

    def _cross_entropy(self, y: int, y_hat: np.ndarray) -> float:
        """
        :param y: integer class label
        :param y_hat: prediction with shape (num_classes,)
        :return: cross entropy loss
        """
        return -np.log(y_hat[y])

    def forward(self, z: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        """
        :param z: input logits of shape (num_classes,)
        :param y: integer class label
        :return:
            y: predictions from softmax as an np.ndarray
            loss: cross entropy loss
        """
        y_hat = self._softmax(z)
        loss = self._cross_entropy(y,y_hat)
        return y_hat, loss

    def backward(self, y: int, y_hat: np.ndarray) -> np.ndarray:
        """
        :param y: integer class label
        :param y_hat: predicted softmax probability with shape (num_classes,)
        :return: gradient with shape (num_classes,)
        """
        y_hat[y] -=1
        return y_hat

class Sigmoid:
    def __init__(self):
        self.sig = None
        return

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: Input to activation function (i.e. output of the previous 
                  linear layer), with shape (output_size,)
        :return: Output of sigmoid activation function with shape
            (output_size,)
        """
        self.sig = 1/(1+np.exp(-x))
        return self.sig

    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        :param dz: partial derivative of loss with respect to output of
            sigmoid activation
        :return: partial derivative of loss with respect to input of
            sigmoid activation
        """
        return dz*(self.sig*(1-self.sig))


INIT_FN_TYPE = Callable[[Tuple[int, int]], np.ndarray]


class Linear:
    def __init__(self, input_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        """
        :param input_size: number of units in the input of the layer 
                           *not including* the folded bias
        :param output_size: number of units in the output of the layer
        :param weight_init_fn: function that creates and initializes weight 
                               matrices for layer. This function takes in a 
                               tuple (row, col) and returns a matrix with
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        """
        # Initialize learning rate for SGD
        self.lr = learning_rate

        self.weight = weight_init_fn((output_size,input_size+1))
        self.weight[:,0] = 0
        self.gradient_w = zero_init((output_size,input_size))
        self.x = None
        self.g_weight = None
        return

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: Input to linear layer with shape (input_size,)
                  where input_size *does not include* the folded bias.
        :return: output z of linear layer with shape (output_size,)
        """

        self.x = np.vstack(([[1]],x))
        return self.weight@self.x

    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        :param dz: partial derivative of loss with respect to output z
            of linear
        :return: dx, partial derivative of loss with respect to input x
            of linear
        """
        self.g_weight = dz@self.x.T
        return self.weight[:,1:].T@dz

    def step(self) -> None:
        """
        Apply SGD update to weights using self.dw, which should have been 
        set in NN.backward().
        """
        self.weight -= self.lr*self.g_weight
        return


class NN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        """
        :param input_size: number of units in input to network
        :param hidden_size: number of units in the hidden layer of the network
        :param output_size: number of units in output of the network - this
                            should be equal to the number of classes
        :param weight_init_fn: function that creates and initializes weight 
                               matrices for layer. This function takes in a 
                               tuple (row, col) and returns a matrix with 
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        """
        self.weight_init_fn = weight_init_fn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.L1 = Linear(input_size,hidden_size,weight_init_fn,learning_rate)
        self.Sigmoid = Sigmoid()
        self.L2 = Linear(hidden_size,output_size,weight_init_fn,learning_rate)
        self.Softmax= SoftMaxCrossEntropy()
        self.loss = 0
        self.y_hat =None
        # print("initialization done")

    def forward(self, x: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        """
        :param x: input data point *without the bias folded in*
        :param y: prediction with shape (num_classes,)
        :return:
            y_hat: output prediction with shape (num_classes,). This should be
                a valid probability distribution over the classes.
            loss: the cross_entropy loss for a given example
        """
        a = self.L1.forward(x.reshape((-1,1)))
        z = self.Sigmoid.forward(a)
        b = self.L2.forward(z)
        y_hat,loss = self.Softmax.forward(b,y)
        # print("y_hat=\n",y_hat)
        # print("loss=\n",loss)
        return y_hat,loss

    def backward(self, y: int, y_hat: np.ndarray) -> None:
        """
        :param y: label (a number or an array containing a single element)
        :param y_hat: prediction with shape (num_classes,)
        """
        gb = self.Softmax.backward(y,y_hat)
        gz = self.L2.backward(gb)
        ga = self.Sigmoid.backward(gz)
        gx = self.L1.backward(ga)
        # print("gb,gz,ga,gx")
        # print(gb,gz,ga,gx)
        # print("gbeta,galpha")
        # print(self.L2.g_weight,self.L1.g_weight)
        return None

    def step(self):
        self.L1.step()
        self.L2.step()
        return None

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        :param X: Input dataset of shape (num_points, input_size)
        :param y: Input labels of shape (num_points,)
        :return: Mean cross entropy loss
        """
        CEL = 0
        N = X.shape[0]
        for i in range(N):
            _,loss = self.forward(X[i,:].T,y[i])
            CEL += loss

        return CEL/N

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              n_epochs: int) -> Tuple[List[float], List[float]]:
        """
        :param X_tr: train data
        :param y_tr: train label
        :param X_test: train data
        :param y_test: train label
        :param n_epochs: number of epochs to train for
        :return:
            train_losses: Training losses *after* each training epoch
            test_losses: Test losses *after* each training epoch
        """
        train_losses = []
        test_losses = []
        N_tr = X_tr.shape[0]
        for epoch in range(n_epochs):
            X,y = shuffle(X_tr,y_tr,epoch)
            for i in range(N_tr):
                # print(y[i])
                y_hat,loss = self.forward(X[i,:],y[i])
                self.backward(y[i],y_hat)
                self.step()
                # print("update")
                # print(self.L1.weight)
                # print(self.L2.weight)
            train_losses.append(self.compute_loss(X_tr,y_tr))
            test_losses.append(self.compute_loss(X_test,y_test))

        return train_losses,test_losses

    def test(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the label and error rate.
        :param X: input data
        :param y: label
        :return:
            labels: predicted labels
            error_rate: prediction error rate
        """
        N = X.shape[0]
        labels = []
        for i in range(N):
            y_hat,_ = self.forward(X[i,:],y[i])
            label = np.argmax(y_hat)
            labels.append(label)

        labels = np.array(labels)
        correct = np.sum(labels==y)
        return labels, 1-correct/N


if __name__ == "__main__":
    args = parser.parse_args()
    # Define our labels
    labels = ["0","1"]

    (X_tr, y_tr, X_test, y_test, out_tr, out_te, out_metrics,
     n_epochs, n_hid, init_flag, lr) = args2data(args)
    #preset
    # labels = ["a","b"]
    # X_tr = np.array([[1,2,3],[2,1,1]]).reshape(2,3)
    # y_tr = np.array([[0],[1]]).reshape(2,1)
    # n_hid = 4
    # init_flag = 1
    # lr = 1
    # X_test = np.array([[1,2,3],[2,1,1]]).reshape(2,3)
    # y_test = np.array([[0],[1]]).reshape(2,1)
    # n_epochs =1

    nn = NN(
        input_size=X_tr.shape[-1],
        hidden_size=n_hid,
        output_size=len(labels),
        weight_init_fn=zero_init if init_flag == 2 else random_init,
        learning_rate=lr
    )

    # train model
    train_losses, test_losses = nn.train(X_tr, y_tr, X_test, y_test, n_epochs)
    train_labels, train_error_rate = nn.test(X_tr, y_tr)
    test_labels, test_error_rate = nn.test(X_test, y_test)

    with open(out_tr, "w") as f:
        for label in train_labels:
            f.write(str(label) + "\n")
    with open(out_te, "w") as f:
        for label in test_labels:
            f.write(str(label) + "\n")
    with open(out_metrics, "w") as f:
        for i in range(len(train_losses)):
            cur_epoch = i + 1
            cur_tr_loss = train_losses[i]
            cur_te_loss = test_losses[i]
            f.write("epoch={} crossentropy(train): {}\n".format(
                cur_epoch, cur_tr_loss))
            f.write("epoch={} crossentropy(validation): {}\n".format(
                cur_epoch, cur_te_loss))
        f.write("error(train): {}\n".format(train_error_rate))
        f.write("error(validation): {}\n".format(test_error_rate))
