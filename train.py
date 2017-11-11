from __future__ import absolute_import
from __future__ import print_function


import sys
from random import shuffle
import numpy as np
from sklearn import metrics
import torch
from helper import add_gradient_noise, exp_lr_scheduler

if "../" not in sys.path:
  sys.path.append("../")


GLOBAL_STEP = 0




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
    update_train = None
    update_eval = None
    # params
    n_train = dataset.trainS.shape[0]
    n_test = dataset.testS.shape[0]
    n_val = dataset.valS.shape[0]

    print("Training Size", n_train)
    print("Validation Size", n_val)
    print("Testing Size", n_test)

    batches = list(zip(range(0, n_train - args.batch_size, args.batch_size),
                  range(args.batch_size, n_train, args.batch_size)))

    # train_labels = np.argmax(dataset.trainA, axis=1)
    # test_labels = np.argmax(dataset.testA, axis=1)
    # val_labels = np.argmax(dataset.valA, axis=1)

    for t in range(args.epochs):
        running_loss = 0.0

        shuffle(batches)
        train_preds = np.array([])
        train_labels = np.array([])
        # optimizer = exp_lr_scheduler(optimizer, t)
        # for i, start in enumerate(range(0, n_train, args.batch_size)):
        #     end = start + args.batch_size
        for i, (start, end) in enumerate(batches):
            assert end - start == args.batch_size
            s = dataset.trainS[start:end]
            q = dataset.trainQ[start:end]
            a = np.argmax(dataset.trainA[start:end], axis=-1)

            optimizer.zero_grad()
            loss, predictions, probs = model.forward(s, s, q, a)
            loss.backward()

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
            # Add gradient noise
            for p in model.parameters():
                p = add_gradient_noise(p)

            optimizer.step()
            running_loss += loss.data[0]
            train_preds = np.concatenate((train_preds, predictions.data.numpy()))
            train_labels = np.concatenate((train_labels, a))
            GLOBAL_STEP += 1

        train_acc = metrics.accuracy_score(train_labels, train_preds)
        vis.line(Y=np.array([running_loss]),
                 X=np.array([t]),
                 opts=dict(legend=["loss",],
                           title="KVMemNet training loss",
                           showlegend=True),
                 win="KVMemNet_training_loss_{}".format(exp_name),
                 update=update_train)
        vis.line(Y=np.array([train_acc]),
                 X=np.array([t]),
                 opts=dict(legend=["accuracy"],
                           title="KVMemNet accuracy",
                           showlegend=True),
                 win="KVMemNet_training_acc_{}".format(exp_name),
                 update=update_train)
        update_train = "append"
        print('-----------------------')
        print('Epoch', t+1)
        print('Loss:', running_loss)
        print('Training Accuracy: {0:.2f}'.format(train_acc))
        print('-----------------------')


        if t % args.evaluation_interval == 0:
            s = dataset.valS
            q = dataset.valQ
            a = np.argmax(dataset.valA, axis=1)

            val_loss, predictions, probs = model.forward(s, s, q, a)
            val_loss = val_loss.data.numpy()[0]
            val_acc = metrics.accuracy_score(a, predictions.data.numpy())
            print('-----------------------')
            print('Eval Epoch', t+1)
            print('Eval Loss:', val_loss)
            print('Eval Accuracy:', val_acc)
            print('-----------------------')
            vis.line(Y=np.array([[val_loss, val_acc]]),
                     X=np.array([[t, t]]),
                     opts=dict(legend=["eval_loss", "eval_accuracy"],
                               title="KVMemNet evaluation",
                               showlegend=True),
                     win="KVMemNet_evaluation_{}".format(exp_name),
                     update=update_eval)
            update_eval = "append"
