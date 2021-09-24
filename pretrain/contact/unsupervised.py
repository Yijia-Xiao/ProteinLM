from io import UnsupportedOperation
import os
import pickle
import numpy as np
import json
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string
import sys


import torch
import torch.nn as nn
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression

import matplotlib.pyplot as plt

msa_path = './data/CAMEO-trRosettaA2M'
label_path = './data/CAMEO-GroundTruth'


range_ = sys.argv[1]
assert range_ in ['medium', 'long', 'midlong'], "should choose from ['medium', 'long', 'midlong']"

print(f'range_ = {range_}')
if range_ == 'medium':
    range_ = [12, 24]
elif range_ == 'long':
    range_ = [24, 2048]
elif range_ == 'midlong':
    range_ = [12, 2048]

frac = int(sys.argv[2])
print(f'frac = {frac}')

# ratio_ = [0.1, 0.9]
ratio_ = [0.2, 0.8]
print(f'ratio_ = {ratio_}')

def process_one_sample(file_name):
    msa = open(os.path.join(msa_path, file_name + '.a2m'), 'r').read().splitlines()  # .readlines()
    with open(os.path.join(label_path, file_name + '.native.pkl'), 'rb') as f:
        labels = pickle.load(f, encoding="bytes")

    assert msa[0].strip() == labels[b'sequence'].decode()
    # an entry of <0 indicates an invalid distance.
    dist_mat = labels[b'atomDistMatrix'][b'CbCb']
    seq_len = len(dist_mat)
    binary_labels = torch.zeros((seq_len, seq_len), dtype=torch.float).tolist()
    # dist_labels = torch.zeros((seq_len, seq_len), dtype=torch.float).tolist()
    # print(seq_len)
    for row in range(seq_len):
        for col in range(seq_len):
            if dist_mat[row][col] >= 0:
                if dist_mat[row][col] < 8:
                    binary_labels[row][col] = 1
            #     dist_labels[row][col] = float(dist_mat[row][col])
            # elif dist_mat[row][col] < 0:
            #     binary_labels[row][col] = -1
            #     dist_labels[row][col] = -1
    return {
        'name': file_name,
        'msa': msa,
        'binary_labels': binary_labels,
    }


# print(len(names), len(msa_names), len(label_names))
# 129 131 129: label is subset of MSAs

names = [i.split('.')[0] for i in os.listdir(label_path)]
print(names)

# process_one_sample(names[0])

def train_classification_net(data, label):
    net = LogisticRegression(penalty='l1', C=1 / 0.15, solver='liblinear')
    # net = LogisticRegression(penalty='l2', C=1 / 0.15, solver='sag', n_jobs=64)
    net.fit(data, label)
    ret = {}
    ret['net.intercept_'] = net.intercept_
    ret['net.coef_'] = net.coef_
    ret['net.score(X, Y)'] = net.score(data, label)
    ret['net'] = net
    return ret


def train_mlp_net(data, label):
    class MLP(nn.Module):
        def __init__(self, in_feat):
            super().__init__()
            self.in_feat = in_feat
            self.fc = nn.Linear(self.in_feat, 1)
        
        def forward(self, x):
            return self.fc(x)

    # net = LogisticRegression(penalty='l1', C=1 / 0.15, solver='liblinear')
    # net = LogisticRegression(penalty='l2', C=1 / 0.15, solver='sag', n_jobs=64)
    net = MLP()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters())
    batch_size = 16
    for i in range(0, len(data), batch_size):
        optimizer.zero_grad()
        batch_data = data[i: i + batch_size]
        batch_labe = label[i: i + batch_size]
        pred = net(batch_data)
        loss = loss_fn(batch_data.reshape(batch_labe.shape), batch_labe)
        loss.backward()
        optimizer.step()

    ret = {}
    ret['net'] = net
    return ret

def esm_heads_predict(MSA: List):
    import sys
    sys.path.append('./esm/')
    import esm
    from esm import Alphabet
    # torch.set_grad_enabled(False)

    # This is an efficient way to delete lowercase characters and insertion characters from a string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)


    def read_sequence(filename: str) -> Tuple[str, str]:
        """ Reads the first (reference) sequences from a fasta or MSA file."""
        record = next(SeqIO.parse(filename, "fasta"))
        return record.description, str(record.seq)


    def remove_insertions(sequence: str) -> str:
        """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
        return sequence.translate(translation)


    def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
        """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
        return [(record.description, remove_insertions(str(record.seq)))
                for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]
    
    model = torch.load('/workspace/pts/msa_transformer.pt').cuda()
    msa_alphabet = Alphabet.from_architecture('msa_transformer')
    msa_batch_converter = msa_alphabet.get_batch_converter()

    # esm_attention_map = []
    with torch.no_grad():
        input_msa = [('', seq) for seq in MSA[:128]]
        # input_msa = [('', seq) for seq in MSA[:64]]
        msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(input_msa)
        msa_batch_tokens = msa_batch_tokens.cuda()
        output = model(msa_batch_tokens, need_head_weights=True)['row_attentions']
    return output[0]


# ratio=[1, 2, 5]
# def calculate_precision(name, pred, label, frac=5, ignore_index=-1, range_=12):
#     "output is regression + attention head"
#     # print(pred.shape, label.shape)
#     visualize(name, pred, label)

#     for i in range(len(label)):
#         for j in range(len(label)):
#             if (abs(i - j) <= 12):
#                 label[i][j] = ignore_index

#     correct = 0
#     total = 0

#     predictions = pred.reshape(-1)
#     labels = label.reshape(-1)

#     valid_masks = (labels != ignore_index)
#     # probs = predictions[:, 1]
#     confidence = predictions
#     valid_masks = valid_masks.type_as(confidence)
#     masked_prob = (confidence * valid_masks).view(-1)
#     seq_len = int(len(labels) ** 0.5)
#     most_likely = masked_prob.topk(seq_len // frac, sorted=False)
#     selected = labels.view(-1).gather(0, most_likely.indices)
#     selected[selected < 0] = 0
#     correct += selected.sum().long()
#     total += selected.numel()
#     return correct, total


def calculate_supervised_precision(name, pred, label, frac=5, ignore_index=-1):
    "output is regression + attention head"
    # visualize(name, pred, label)
    # print(pred.shape, label.shape)
    
    for i in range(len(label)):
        for j in range(len(label)):
            if (abs(i - j) < range_[0] or abs(i - j) >= range_[1]):
                label[i][j] = ignore_index

    correct = 0
    total = 0

    predictions = pred
    labels = label.reshape(-1)

    valid_masks = (labels != ignore_index)
    confidence = predictions[:, 1]
    # confidence = predictions
    valid_masks = valid_masks.type_as(confidence)
    masked_prob = (confidence * valid_masks).view(-1)
    seq_len = int(len(labels) ** 0.5)
    most_likely = masked_prob.topk(seq_len // frac, sorted=False)
    selected = labels.view(-1).gather(0, most_likely.indices)
    selected[selected < 0] = 0
    correct += selected.sum().long()
    total += selected.numel()
    return correct, total


def visualize(name, pred, label):
    pred, label = pred.cpu(), label.cpu()
    fig, axes = plt.subplots(figsize=(18, 6), ncols=2)
    seqlen = len(label)
    axes[0].imshow(pred[:seqlen, :seqlen], cmap="Blues")
    axes[1].imshow(label[:seqlen, :seqlen], cmap="Blues")
    plt.savefig(f'./vis/{name}.png')


def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)


def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized


# def eval_unsupervised(ratio=[0.2, 0.8]):
# def eval_unsupervised(ratio=[0.3, 0.7]):
def eval_unsupervised(ratio=ratio_):
    correct = 0
    total = 0
    data = []

    # train_samples = int(ratio[0] * len(names))
    train_samples = 5
    test_samples = int(ratio[1] * len(names))
    esm_train = []
    bin_train = []
    # train
    def construct_train(heads, bin_label):
        # with -1 for invalid flag
        # print(bin_label)
        # bin_label_with_invalid = torch.from_numpy(np.array(bin_label))
        # print(f'-1 {(bin_label_with_invalid == -1).sum()}')
        # bin_label_without_invalid = torch.max(bin_label_with_invalid, 0)
        # print(bin_label_without_invalid)

        # heads = heads.permute(2, 3, 0, 1).reshape(-1, 144)  # .transpose(0, 1)
        num_layer, num_head, seqlen, _ = heads.size()
        attentions = heads.view(num_layer * num_head, seqlen, seqlen)
        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(1, 2, 0)
        # print(attentions.shape) # torch.Size([383, 383, 144])
        # print(bin_label.shape)
        # for pixel, p_label in zip(heads, bin_label_without_invalid):
        #     esm_train.append(pixel.tolist())
        #     bin_train.append(p_label.item())
        for i in range(seqlen):
            for j in range(seqlen):
                # if abs(i - j) >= range_:
                if (abs(i - j) >= range_[0] and abs(i - j) < range_[1]):
                    esm_train.append(attentions[i][j].tolist())
                    bin_train.append(bin_label[i][j].item())

    for name in names[: train_samples]:
        data.append(process_one_sample(name))
        if len(data[-1]['msa'][0]) > 1024:
            print(f'skipped one sample with length {len(data[-1]["msa"][0])}')
            continue
        heads = esm_heads_predict(data[-1]['msa'])[:, :, 1:, 1:]
        label = torch.from_numpy(np.array(data[-1]['binary_labels']))
        construct_train(heads, label)

    print('start train')
    net = train_classification_net(esm_train, bin_train)
    print('stop train')
    print(net)

    # test
    for name in names[train_samples: train_samples + test_samples]:
        data.append(process_one_sample(name))
        if len(data[-1]['msa'][0]) > 1024:
            print(f'skipped one sample with length {len(data[-1]["msa"][0])}')
            continue
        heads = esm_heads_predict(data[-1]['msa'])[:, :, 1:, 1:]
        label = torch.from_numpy(np.array(data[-1]['binary_labels']))

        num_layer, num_head, seqlen, _ = heads.size()
        attentions = heads.view(num_layer * num_head, seqlen, seqlen)
        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(1, 2, 0)

        proba = net['net'].predict_proba(attentions.reshape(-1, 144).cpu())
        # print(proba)
        cor, tot = calculate_supervised_precision(data[-1]['name'], torch.from_numpy(proba).to('cuda'), label.to('cuda'), frac=frac)
        correct += cor
        total += tot
        print(cor.item(), tot)

    print(f'correct = {correct.item()}, total = {total}, precision = {correct.item() / total}')

eval_unsupervised()

"""
def eval_unsupervised():
    correct = 0
    total = 0
    data = []
    for name in names:
        data.append(process_one_sample(name))
        # esm_predict(data[-1]['msa'])
        # print(len(data[-1]['msa'][0]))
        if len(data[-1]['msa'][0]) > 1024:
            print(f'skipped one sample with length {len(data[-1]["msa"][0])}')
            continue
        pred = unsupervised_esm_predict(data[-1]['msa'])
        label = torch.from_numpy(np.array(data[-1]['binary_labels']))
        cor, tot = calculate_precision(data[-1]['name'], pred.to('cuda'), label.to('cuda'), frac=1)
        correct += cor
        total += tot
        print(cor.item(), tot)
        # break

    print(f'correct = {correct.item()}, total = {total}, precision = {correct.item() / total}')



"""