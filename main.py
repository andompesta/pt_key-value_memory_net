from __future__ import absolute_import
from __future__ import print_function

import argparse
from datetime import datetime
from model_KVMemNet import KVMemNet
from torch import optim
from visdom import Visdom
from helper import use_cuda, vectorize_data, load_task, Dataset
import numpy as np
from functools import reduce
from itertools import chain
from sklearn import model_selection
from train import work

EXP_NAME = "exp-{}".format(datetime.now())


def __pars_args__():
    parser = argparse.ArgumentParser(description='KV_MemNet')
    parser.add_argument('--seed', type=int, default=905, help='random seed (default: 1)')

    parser.add_argument('--task_id', type=int, default=20, help="Task to execute.")
    parser.add_argument("--data_dir", type=str, default="dataset/en-10k/", help="Directory containing bAbI tasks")
    parser.add_argument("--output_file", type=str, default="single_scores.csv", help="Name of output file for final bAbI accuracy scores.")

    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for Adam Optimizer.')
    parser.add_argument('-l2', '--l2_lambda', type=float, default=0.1, help='Lambda for l2 loss.')
    parser.add_argument('--max_grad_norm', type=float, default=20.0, help="Clip gradients to this norm.")
    parser.add_argument('--keep_prob', type=float, default=1., help="Keep probability for dropout.")
    parser.add_argument("--evaluation_interval", type=int, default=40, help="Evaluate and print results every x epochs")
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training.')
    parser.add_argument('--feature_size', type=int, default=50, help='Feature size.')
    parser.add_argument('--hops', type=int, default=3, help='Number of hops in the Memory Network.')
    parser.add_argument('--epochs', type=int, default=200, help="Number of epochs to train for.")
    parser.add_argument('--embedding_size', type=int, default=40, help="Embedding size for embedding matrices.")
    parser.add_argument('--memory_size', type=int, default=50, help="Maximum size of memory.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--train', default=True, help='if we want to update the master weights')
    return parser.parse_args()


def __dataset_preparation__(args, train, test):
    """
    Preparation of training, validation and testing set
    :param args: argumets
    :param train: training set
    :param test: testing set
    :return:
    """
    data = train + test

    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    vocab_size = len(word_idx) + 1  # +1 for nil word


    # data analysis
    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean(list(map(len, (s for s, _, _ in data)))))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    query_size = max(map(len, (q for _, q, _ in data)))
    memory_size = min(args.memory_size, max_story_size)
    sentence_size = max(query_size, sentence_size)  # for the position

    print("Longest sentence length", sentence_size)
    print("Longest story length", max_story_size)
    print("Average story length", mean_story_size)

    # train/validation/test sets
    S, Q, A = vectorize_data(train, word_idx, sentence_size, memory_size)
    trainS, valS, trainQ, valQ, trainA, valA = model_selection.train_test_split(S, Q, A, test_size=.1,
                                                                                random_state=args.seed)
    testS, testQ, testA = vectorize_data(test, word_idx, sentence_size, memory_size)

    dataset = Dataset(trainS, valS, testS, trainQ, valQ, testQ, trainA, valA, testA)
    return vocab, word_idx, vocab_size, memory_size, sentence_size, query_size, dataset

if __name__ == '__main__':
    args = __pars_args__()
    print("Started Task:", args.task_id)

    train, test = load_task(args.data_dir, args.task_id)
    vocab, word_idx, vocab_size, memory_size, sentence_size, query_size, dataset = __dataset_preparation__(args, train, test)

    viz = Visdom()
    assert viz.check_connection()

    model = KVMemNet(batch_size=args.batch_size,
                     embedding_size=args.embedding_size,
                     memory_size=memory_size,
                     vocab_size=vocab_size,
                     story_size=sentence_size,
                     query_size=sentence_size,
                     keep_prob=args.keep_prob,
                     feature_size=args.feature_size,
                     l2_lambda=args.l2_lambda,
                     hops=args.hops)
    if use_cuda:
        model.cuda()

    optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate, eps=args.epsilon)
    work(model, dataset, args, viz, EXP_NAME, optimizer)