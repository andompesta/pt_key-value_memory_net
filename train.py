from __future__ import absolute_import
from __future__ import print_function


import sys
from random import shuffle
import numpy as np
from sklearn import model_selection, metrics
import torch
from torch.autograd import Variable
import helper as helper
from itertools import chain
from functools import reduce

if "../" not in sys.path:
  sys.path.append("../")


GLOBAL_STEP = 0


def train_step(model, s, q, a, optimizer):
    # zero the parameter gradients
    optimizer.zero_grad()

    loss, predictions, probs = model.forward(s, s, q, a)
    loss.backward()
    optimizer.step()

    return predict_op




def work(model, dataset, args, vis, exp_name, optimizer):
    """
    Train the model
    :param network: network to train
    :param args: args to be used
    :param vis: Visdom server
    :param exp_name: experiment name
    :param optimizer: optimizer used
    :return:
    """
    global GLOBAL_STEP

    # params
    n_train = dataset.trainS.shape[0]
    n_test = dataset.testS.shape[0]
    n_val = dataset.valS.shape[0]

    print("Training Size", n_train)
    print("Validation Size", n_val)
    print("Testing Size", n_test)

    batches = list(zip(range(0, n_train - args.batch_size, args.batch_size),
                  range(args.batch_size, n_train, args.batch_size)))

    train_labels = np.argmax(dataset.trainA, axis=1)
    test_labels = np.argmax(dataset.testA, axis=1)
    val_labels = np.argmax(dataset.valA, axis=1)

    for t in range(1, args.epochs + 1):
        running_loss = 0.0

        shuffle(batches)
        train_preds = []
        for i, (start, end) in enumerate(batches):
            assert end - start == args.batch_size
            s = dataset.trainS[start:end]
            q = dataset.trainQ[start:end]
            a = dataset.trainA[start:end]

            optimizer.zero_grad()
            loss, predictions, probs = model.forward(s, s, q, a)
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
            optimizer.step()
            # TODO: add gradient noise
            running_loss += loss.data[0]

            train_preds += list(predictions)
            if i % 10 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (t, i + 1, running_loss))

        train_acc = metrics.accuracy_score(train_labels, train_preds)
        print('-----------------------')
        print('Epoch', t)
        print('Training Accuracy: {0:.2f}'.format(train_acc))
        print('-----------------------')

        if t % args.evaluation_interval == 0:
            s = dataset.valS[start:end]
            q = dataset.valQ[start:end]
            a = dataset.valA[start:end]

            loss, predictions, probs = model.forward(s, s, q, a)
            val_acc = metrics.accuracy_score(val_labels, predictions)
            print('-----------------------')
            print('Epoch', t)
            print('Validation Accuracy:', val_acc)
            print('-----------------------')
            # test on train dataset
