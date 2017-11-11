import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from helper import TENSOR_TYPE, AttrProxy
import math


def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = torch.ones(embedding_size, sentence_size)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return torch.t(encoding)


class KVMemNet(nn.Module):
    def __init__(self,  batch_size, embedding_size, memory_size, vocab_size, story_size, query_size,
                 keep_prob=1, feature_size=30, l2_lambda=0.2, hops=3):
        """
        Key-value model
        :param embedding_size: the size of the word embedding
        :param memory_size: size of the key-value memory
        :param feature_size: dimension of feature extraced from word embedding
        :param reader: type of reader of the content
        :param l2_lambda: l2 regulariser
        """
        super(KVMemNet, self).__init__()
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.memory_size = memory_size
        self.vocab_size = vocab_size
        self.story_size = story_size
        self.query_size = query_size
        self.feature_size = feature_size
        self.l2_lambda = l2_lambda
        self.encoding = Variable(position_encoding(self.story_size, self.embedding_size), requires_grad=False, volatile=False)
        self.keep_prob = keep_prob
        self.hops = hops


        self.A = nn.Parameter(TENSOR_TYPE['f_tensor'](self.feature_size, self.embedding_size))
        self.TK = nn.Parameter(TENSOR_TYPE['f_tensor'](self.memory_size, self.embedding_size))
        self.TV = nn.Parameter(TENSOR_TYPE['f_tensor'](self.memory_size, self.embedding_size))

        self.W = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        self.W_memory = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        self.Rs = nn.ParameterList([nn.Parameter(TENSOR_TYPE['f_tensor'](self.feature_size, self.feature_size)) for _ in range(self.hops)])

        self._loss = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.xavier_uniform(p)

    def forward(self, memory_key, memory_value, query, labels):

        memory_key = Variable(torch.from_numpy(memory_key))
        memory_value = Variable(torch.from_numpy(memory_value))
        query = Variable(torch.from_numpy(query))
        labels = Variable(torch.from_numpy(labels))

        embedded_chars = self.W(query)                                # shape: [batch_size, query_size, embedding_size]

        memory_shape = memory_key.data.shape
        mkeys_embedded_chars = self.W_memory(memory_key.view(memory_shape[:-2] + torch.Size([-1])))              # shape: [batch_size, memory_size, story_size, embedding_size]
        mvalues_embedded_chars = self.W_memory(memory_value.view(memory_shape[:-2] + torch.Size([-1])))          # shape: [batch_size, memory_size, story_size, embedding_size]
        mkeys_embedded_chars = mkeys_embedded_chars.view(memory_shape + torch.Size([-1]))
        mvalues_embedded_chars = mvalues_embedded_chars.view(memory_shape + torch.Size([-1]))

        q_r = torch.sum((embedded_chars * self.encoding), dim=1)
        doc_r = torch.sum((mkeys_embedded_chars * self.encoding), dim=2)
        value_r = torch.sum((mvalues_embedded_chars * self.encoding), dim=2)
        o = self.__forward_loop__(doc_r, value_r, q_r)
        o = torch.t(o).contiguous()

        y_tmp = torch.mm(self.A, torch.t(self.W_memory.weight))
        logits = torch.mm(o, y_tmp)  # + logits_bias
        probs = F.softmax(logits)

        cross_entropy = self._loss(logits, labels)
        _, predict_op = torch.max(probs, dim=1)
        return cross_entropy, predict_op, probs


    def __forward_loop__(self, mkeys, mvalues, questions):
        """
        perform a learning loop
        :param mkeys: memory key representation
        :param mvalues: memory value representation
        :param questions: question representation
        :return:
        """
        u = torch.mm(self.A, torch.t(questions))                    # [feature_size, batch_size]
        u = [u]
        for _, R in enumerate(self.Rs):
            u_temp = u[-1]
            mk_temp = mkeys + self.TK                                                                   # [batch_size,  memory_size, embedding_size]
            k_temp = mk_temp.permute(2, 0, 1).contiguous().view(self.embedding_size, -1)                # [embedding_size, batch_size * memory_size]
            a_k_temp = torch.mm(self.A, k_temp)                                                         # [feature_size, batch_size x memory_size]
            a_k = torch.t(a_k_temp).contiguous().view(-1, self.memory_size, self.feature_size)          # [batch_size, memory_size, feature_size]
            u_expanded = torch.t(u_temp).contiguous().unsqueeze(dim=1)                                  # [batch_size, 1, feature_size]
            dotted = torch.sum((a_k * u_expanded), dim=2)                                               # [batch_size, memory_size]

            # Calculate probabilities
            probs = F.softmax(dotted)                                                           # [batch_size, memory_size]
            probs_expand = probs.unsqueeze(dim=-1)                                                      # [batch_size, memory_size, 1]
            mv_temp = mvalues + self.TV                                                                 # [batch_size,  memory_size, embedding_size]
            v_temp = mv_temp.permute(2, 0, 1).contiguous().view(self.embedding_size, -1)                # [embedding_size, batch_size x memory_size]
            a_v_temp = torch.mm(self.A, v_temp)                                                         # [feature_size, batch_size x memory_size]
            a_v = torch.t(a_v_temp).contiguous().view(-1, self.memory_size, self.feature_size)          # [batch_size, memory_size, feature_size]

            o_k = torch.t(torch.sum(probs_expand * a_v, dim=1))                                         # [feature_size, batch_size]
            u_k = torch.mm(R, (u[-1] + o_k))                                                           # [feature_size, batch_size]
            u.append(u_k)
        return u[-1]



