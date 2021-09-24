import os
import sys
import pickle
import numpy as np
import random
import json
from scipy.spatial.distance import pdist, squareform
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string
import torch
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression

import matplotlib.pyplot as plt

# eg:
    # python prepare_trRosetta.py midlong 5

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


# (alphabet 'ARNDCQEGHILKMFPSTWYV-')
alphabet_str = 'ARNDCQEGHILKMFPSTWYV-'
id_to_char = dict()
for i in range(len(alphabet_str)):
    id_to_char[i] = alphabet_str[i]


def map_id_to_token(token_id):
    return id_to_char[token_id]


def prepare_data(num_samples=20):
    ret_data = []
    npz_path = "./data/trRosetta/"
    files = os.listdir(npz_path)
    for f in files[: num_samples]:
        abs_path = os.path.join(npz_path, f)
        data = np.load(abs_path)
        msa_ids = data['msa']
        xyzCa = data['xyzca']
        msa_ids_2D_list = msa_ids.tolist()
        msa = []
        for seq in msa_ids_2D_list:
            msa_str = ''.join(list(map(map_id_to_token, seq)))
            # print(msa_str)
            msa.append(msa_str)
        contact = np.less(squareform(pdist(xyzCa)), 8.0).astype(np.int64)
        # print(contact, contact.shape)
        sample = {'name': f, 'msa': msa, 'contact': contact}
        # print(msa_ids.shape, len(msa), len(msa[0]))
        ret_data.append(sample)
    return np.array(ret_data, dtype=object)


# data = prepare_data(20)
# np.save('./data/trRosetta', data)


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
        msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(input_msa)
        msa_batch_tokens = msa_batch_tokens.cuda()
        output = model(msa_batch_tokens, need_head_weights=True)['row_attentions']
    return output[0]


def generate_esm_pred():
    data = np.load('./data/trRosetta.npy', allow_pickle=True)
    for idx, sample in enumerate(data):
        msa = sample['msa']
        esm_pred = esm_heads_predict(msa)
        # print(esm_pred.shape, np.array(sample['contact']).shape)
        # return
        data[idx]['esm'] = esm_pred
    np.save('./data/trRosetta-esm', data)

# generate_esm_pred()


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


def process_one_sample(file_name):
    msa_path = './data/CAMEO-trRosettaA2M'
    label_path = './data/CAMEO-GroundTruth'
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
                # dist_labels[row][col] = float(dist_mat[row][col])
            else:
                binary_labels[row][col] = -1
                # dist_labels[row][col] = -1
    return {
        'name': file_name,
        'msa': msa,
        'binary_labels': binary_labels,
    }


def eval_unsupervised():
    preds_and_label = np.load('./data/trRosetta-esm.npy', allow_pickle=True) # [:1]

    correct = 0
    total = 0
    data = []

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
                    bin_train.append(bin_label[i][j])

    # for name in names[: train_samples]:
    for sample in preds_and_label:
        # data.append(process_one_sample(name))
        # if len(data[-1]['msa'][0]) > 1024:
        #     print(f'skipped one sample with length {len(data[-1]["msa"][0])}')
        #     continue
        # heads = esm_heads_predict(data[-1]['msa'])[:, :, 1:, 1:]
        # label = torch.from_numpy(np.array(data[-1]['binary_labels']))
        heads = sample['esm'][:, :, 1:, 1:]
        label = sample['contact']
        # print(heads.shape, len(label), len(label[0])) # torch.Size([12, 12, 155, 155]) 154 154
        construct_train(heads, label)
        # print(len(esm_train))
        # 20306
        # 137612
        # 215174

    print('start train')
    net = train_classification_net(esm_train, bin_train)
    print('stop train')
    print(net)

    # test
    label_path = './data/CAMEO-GroundTruth'
    names = [i.split('.')[0] for i in os.listdir(label_path)]
    data = []
    for name in names:
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


def move_files():
    npz_path = "/workspace/training_set/npz"
    files = os.listdir(npz_path)
    random.shuffle(files)
    for f in files[:100]:
        abs_path = os.path.join(npz_path, f)
        os.system(f'cp {abs_path} data/trRosetta/')

# move_files()