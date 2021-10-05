import os
import sys
import pickle
import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import List, Tuple
import torch
from sklearn.linear_model import Lasso, ElasticNet, LogisticRegression

DATA_ROOT = './data/contact-data/'

import sys
msa_depth = int(sys.argv[1])
sklearn_solver = sys.argv[2]
assert sklearn_solver in ['liblinear', 'saga'], 'sklearn solver error'

alphabet_str = 'ARNDCQEGHILKMFPSTWYV-'
id_to_char = dict()

for i in range(len(alphabet_str)):
    id_to_char[i] = alphabet_str[i]


def map_id_to_token(token_id):
    return id_to_char[token_id]


def prepare_megatron():
    def process_trRosetta():
        os.system(f'rm {DATA_ROOT}/megatron/train.json')
        ret_data = []
        npz_path = f"{DATA_ROOT}/train20/"
        files = os.listdir(npz_path)
        for f in files:
            abs_path = os.path.join(npz_path, f)
            data = np.load(abs_path)
            msa_ids = data['msa']
            xyzCa = data['xyzca']
            msa_ids_2D_list = msa_ids.tolist()
            msa = []
            for seq in msa_ids_2D_list:
                msa_str = ''.join(list(map(map_id_to_token, seq)))
                msa.append(msa_str)
            contact = np.less(squareform(pdist(xyzCa)), 8.0).astype(np.int64)
            sample = {'name': f, 'msa': msa, 'contact': contact}
            with open(f'{DATA_ROOT}/megatron/train.json', 'a+') as f:
                one_seq = sample['msa'][0] + '|' + ''.join(sample['msa'][1:])
                one_seq = ' '.join(one_seq)
                f.write('{"text": ' + '"{}"}}'.format(one_seq) + "\n")
            ret_data.append(sample)
        return np.array(ret_data, dtype=object)

    def process_one_sample(file_name):
        msa_path = f'{DATA_ROOT}/CAMEO-trRosettaA2M'
        label_path = f'{DATA_ROOT}/CAMEO-GroundTruth'
        msa = open(os.path.join(msa_path, file_name + '.a2m'), 'r').read().splitlines()  # .readlines()
        with open(os.path.join(label_path, file_name + '.native.pkl'), 'rb') as f:
            labels = pickle.load(f, encoding="bytes")

        assert msa[0].strip() == labels[b'sequence'].decode()
        # an entry of <0 indicates an invalid distance.
        dist_mat = labels[b'atomDistMatrix'][b'CbCb']
        seq_len = len(dist_mat)
        binary_labels = torch.zeros((seq_len, seq_len), dtype=torch.float).tolist()
        for row in range(seq_len):
            for col in range(seq_len):
                if dist_mat[row][col] >= 0:
                    if dist_mat[row][col] < 8:
                        binary_labels[row][col] = 1
                else:
                    binary_labels[row][col] = -1
        return {
            'name': file_name,
            'msa': msa,
            'binary_labels': binary_labels,
        }

    def process_CAMEO():
        trRosetta_data = []
        label_path = f'{DATA_ROOT}/CAMEO-GroundTruth'
        names = [i.split('.')[0] for i in os.listdir(label_path)]
        for name in names:
            data = process_one_sample(name)
            # print(data['msa'])
            with open(f'{DATA_ROOT}/megatron/test.json', 'a+') as f:
                one_seq = data['msa'][0] + '|' + ''.join(data['msa'][1:])
                one_seq = ' '.join(one_seq)
                f.write('{"text": ' + '"{}"}}'.format(one_seq) + "\n")

            trRosetta_data.append(data)

        return np.array(trRosetta_data, dtype=object)

    train = process_trRosetta()
    print(len(train))
    np.save(f'{DATA_ROOT}/megatron/train_dataset', train, allow_pickle=True)

    test = process_CAMEO()
    print(len(test))
    np.save(f'{DATA_ROOT}/megatron/test_dataset', test, allow_pickle=True)


# prepare_megatron()
# output: 20, 129
# exit(0)

def build_data():
    tasks = ['train', 'test']
    cmd = """/opt/conda/bin/python ../tools/preprocess_data.py --input ./data/megatron/{}.json \
        --tokenizer-type BertWordPieceCase --vocab-file ../msa_tools/msa_vocab.txt \
        --output-prefix ./data/megatron/{} --dataset-impl mmap --workers 20"""
    # for i in tasks:
    #     os.system(cmd.format(i ,i))
    print(cmd.format(tasks[0], tasks[0]))
    print(cmd.format(tasks[1], tasks[1]))

# build_data()
# exit(0)
# /opt/conda/bin/python ../tools/preprocess_data.py --input ./data/megatron/train.json         --tokenizer-type BertWordPieceCase --vocab-file ../msa_tools/msa_vocab.txt         --output-prefix ./data/megatron/train --dataset-impl mmap --workers 20
# /opt/conda/bin/python ../tools/preprocess_data.py --input ./data/megatron/test.json         --tokenizer-type BertWordPieceCase --vocab-file ../msa_tools/msa_vocab.txt         --output-prefix ./data/megatron/test --dataset-impl mmap --workers 20


def megatron_predict():
    cmd = """CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 7008 /dataset/ee84df8b/release/ProteinLM/pretrain/pretrain_tape.py --num-layers 12 \
        --hidden-size 768 --num-attention-heads 12 --micro-batch-size 1 --global-batch-size 1 --seq-length 1024 --max-position-embeddings 1024 --train-iters 1 \
        --data-path /dataset/ee84df8b/release/ProteinLM/pretrain/contact/data/megatron/{}_text_document --vocab-file /root/release/ProteinLM/pretrain/msa_tools/msa_vocab.txt --data-impl mmap \
        --distributed-backend nccl --lr 0 --log-interval 1 --save-interval 2000 --eval-interval 1 --eval-iters {} --max-tokens 262144 --max-aligns 256 --max-length 1024 \
        --tensor-model-parallel-size 1 --no-scaled-masked-softmax-fusion --override-lr-scheduler --mask-prob 0 --split 0,0,1 --checkpoint-activations --attention-save \
        --attention-name {} --finetune --attention-path /dataset/ee84df8b/release/ProteinLM/pretrain/contact/data/pred-megatron/ \
        --load /workspace/ckpt/release/768h-12l-12hd-1mbs-512gbs-1mp-16384tokens-256aligns-1024length-1600ws-100000iter-release"""
    arg_dict = [['train', 21], ['test', 129]]

    # for arg in arg_dict:
    print(cmd.format(arg_dict[0][0], arg_dict[0][1], arg_dict[0][0]))
    print(cmd.format(arg_dict[1][0], arg_dict[1][1], arg_dict[1][0]))

# megatron_predict()
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 7008 /dataset/ee84df8b/release/ProteinLM/pretrain/pretrain_tape.py --num-layers 12         --hidden-size 768 --num-attention-heads 12 --micro-batch-size 1 --global-batch-size 1 --seq-length 1024 --max-position-embeddings 1024 --train-iters 1         --data-path /dataset/ee84df8b/release/ProteinLM/pretrain/contact/data/megatron/train_text_document --vocab-file /root/release/ProteinLM/pretrain/msa_tools/msa_vocab.txt --data-impl mmap         --distributed-backend nccl --lr 0 --log-interval 1 --save-interval 2000 --eval-interval 1 --eval-iters 21 --max-tokens 262144 --max-aligns 256 --max-length 1024         --tensor-model-parallel-size 1 --no-scaled-masked-softmax-fusion --override-lr-scheduler --mask-prob 0 --split 0,0,1 --checkpoint-activations --attention-save         --attention-name train_256 --finetune --attention-path /dataset/ee84df8b/release/ProteinLM/pretrain/contact/data/pred-megatron/         --load /workspace/ckpt/release/768h-12l-12hd-1mbs-512gbs-1mp-16384tokens-256aligns-1024length-1600ws-100000iter-release
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 7008 /dataset/ee84df8b/release/ProteinLM/pretrain/pretrain_tape.py --num-layers 12         --hidden-size 768 --num-attention-heads 12 --micro-batch-size 1 --global-batch-size 1 --seq-length 1024 --max-position-embeddings 1024 --train-iters 1         --data-path /dataset/ee84df8b/release/ProteinLM/pretrain/contact/data/megatron/test_text_document --vocab-file /root/release/ProteinLM/pretrain/msa_tools/msa_vocab.txt --data-impl mmap         --distributed-backend nccl --lr 0 --log-interval 1 --save-interval 2000 --eval-interval 1 --eval-iters 129 --max-tokens 262144 --max-aligns 256 --max-length 1024         --tensor-model-parallel-size 1 --no-scaled-masked-softmax-fusion --override-lr-scheduler --mask-prob 0 --split 0,0,1 --checkpoint-activations --attention-save         --attention-name test_256 --finetune --attention-path /dataset/ee84df8b/release/ProteinLM/pretrain/contact/data/pred-megatron/         --load /workspace/ckpt/release/768h-12l-12hd-1mbs-512gbs-1mp-16384tokens-256aligns-1024length-1600ws-100000iter-release


def load_predictions():
    train_data = torch.load('/workspace/dump/0_train.pt')[:-13]
    # print(len(train_data))
    test_data = torch.load('/workspace/dump/0_test.pt')
    print(len(train_data))
    print(len(test_data))

# load_predictions()
# exit(0)


class MegatronFake(object):
    def __init__(self) -> None:
        super().__init__()
        num_iter = 48000
        # self.train_data = torch.load(f'/dataset/ee84df8b/release/ProteinLM/pretrain/data/attention/esm_style_train_depth{msa_depth}.pt')[:-13]
        # self.test_data = torch.load(f'/dataset/ee84df8b/release/ProteinLM/pretrain/data/attention/esm_style_test_depth{msa_depth}.pt')
        self.train_data = torch.load(f'/dataset/ee84df8b/release/ProteinLM/pretrain/data/attention/megatron_{num_iter}_train_depth{msa_depth}.pt')[:-13]
        self.test_data = torch.load(f'/dataset/ee84df8b/release/ProteinLM/pretrain/data/attention/megatron_{num_iter}_test_depth{msa_depth}.pt')
        self.train_sample = 0
        self.test_sample = 0

        for idx in range(0, len(self.train_data)):
            if idx % 13 == 0:
                continue
            self.train_data[idx] = self.train_data[idx].float().softmax(dim=-1)

        for idx in range(0, len(self.test_data)):
            if idx % 13 == 0:
                continue
            self.test_data[idx] = self.test_data[idx].float().softmax(dim=-1)

    def train_call(self, input_seq):
        for idx in range(0, len(self.train_data), 13):
            # if input_seq in self.train_data[idx][0]:
            if input_seq in self.train_data[idx][0][0]:
                self.train_sample += 1
                return torch.stack(self.train_data[idx + 1: idx + 13])

    def test_call(self, input_seq):
        for idx in range(0, len(self.test_data), 13):
            if input_seq == self.test_data[idx][0][0]:
                self.test_sample += 1
                return torch.stack(self.test_data[idx + 1: idx + 13])


model = MegatronFake()

# test MSA find
# train_ret = model.train_call("MKTIIINGVQFNTDEDTTILKFARDNNIDISALCFLNNCNNDINKCEICTVEVEGTGLVTACDTLIEDGMIINTNSDAVNEKIKSRISQLLDIHEFKCGPCNRRENCEFLKLVIKYKARASKPFLPKDKTEYVDERSKSLTVDRTKCLLCGRCVNACGKNTETYAMKFLNKNGKTIIGAEDEKCFDDTNCLLCGQCIIACPVAALSEKSHMDRVKNALNAPEKHVIVAMAPSVRASIGELFNMGFGVDVTGKIYTALRQLGFDKIFDINFGADMTIMEEATELVQRIENNGPFPMFTSCCPGWVRQAENYYPELLNNLSSAKSPQQIFGTASKTYYPSISGLDPKNVFTVTVMPCTSKKFEADRPQMEKDGLRDIDAVITTRELAKMIKDAKIPFAKLEDSEADPAMGEYSGAGAIFGATGGVMEAALRSAKDFAENAELEDIEYKQVRGLNGIKEAEVEINNNKYNVAVINGASNLFKFMKSGMINEKQYHFIEVMACHGGCVNGGGQPHVNPKDLEKVDIKKVRASVLYNQDEHLSKRKSHENTALVKMYQNYFGKPGEGRAHEILHFKYKKSAWSHPQF")
# test_ret = model.test_call("MFIENKPGEIELLSFFESEPVSFERDNISFLYTAKNKCGLSVDFSFSVVEGWIQYTVRLHENEILHNSIDGVSSFSIRNDNLGDYIYAEIITKELINKIEIRIRPDIKIKSSSVIR")

# exit(0)


def train_classification_net(data, label):
    if sklearn_solver == 'liblinear':
        net = LogisticRegression(penalty='l1', C=1 / 0.15, solver='liblinear')
    elif sklearn_solver == 'saga':
        net = LogisticRegression(penalty='l1', C=1 / 0.15, solver='saga', n_jobs=32)
    net.fit(data, label)
    ret = {}
    ret['net.intercept_'] = net.intercept_
    ret['net.coef_'] = net.coef_
    ret['net.score(X, Y)'] = net.score(data, label)
    ret['net'] = net
    return ret


def calculate_contact_precision(name, pred, label, local_range, local_frac=5, ignore_index=-1):
    """
        local_range: eg. local_range=[12, 24], calculate midium range contacts
        local_frac: eg. local_frac=5, calculate P@L/5, local_frac=2, calculate P@L/2
    """
    for i in range(len(label)):
        for j in range(len(label)):
            if (abs(i - j) < local_range[0] or abs(i - j) >= local_range[1]):
                label[i][j] = ignore_index

    correct = 0
    total = 0

    predictions = pred
    labels = label.reshape(-1)

    valid_masks = (labels != ignore_index)
    confidence = predictions[:, 1]
    valid_masks = valid_masks.type_as(confidence)
    masked_prob = (confidence * valid_masks).view(-1)
    seq_len = int(len(labels) ** 0.5)
    most_likely = masked_prob.topk(seq_len // local_frac, sorted=False)
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


eval_dic = dict()
range_dic = {'short': [6, 12], 'mid': [12, 24], 'long': [24, 2048], 'midlong': [12, 2048], 'all': [-1, 2048]} # , 'midlong': [12, 2048]}

frac_list = [1, 2, 5]
for r in range_dic:
    eval_dic[r] = dict()
    for f in frac_list:
        eval_dic[r][f] = dict()
        for c in ['cor', 'tot']:
            eval_dic[r][f][c] = 0


def eval_unsupervised():
    testset = np.load(f'{DATA_ROOT}/megatron/test_dataset.npy', allow_pickle=True)

    data = []

    esm_train = []
    bin_train = []

    # train
    def construct_train(heads, bin_label):
        num_layer, num_head, seqlen, _ = heads.size()

        attentions = heads.view(num_layer * num_head, seqlen, seqlen)
        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(1, 2, 0)

        for i in range(seqlen):
            for j in range(i + 6, seqlen):
                # based on ESM paper
                esm_train.append(attentions[i][j].tolist())
                bin_train.append(bin_label[i][j])

    import pickle
    trainset = np.load(f'{DATA_ROOT}/megatron/train_dataset.npy', allow_pickle=True)
    for sample in trainset:
        msa = sample['msa'][0]
        label = sample['contact']
        heads = model.train_call(msa)[:, :, 1:, 1:]
        construct_train(heads, label)
    print('start train')
    print(f'{model.train_sample=}')
    net = train_classification_net(esm_train, bin_train)
    print('stop train')
    print(net)

    with open('net.pickle', 'wb') as f:
        pickle.dump(net, f)
    # with open('net.pickle', 'rb') as f:
    #     net = pickle.load(f)

    data = []
    print('start eval')
    for sample in testset:
        data.append(sample)
        if len(data[-1]['msa'][0]) > 1024:
            print(f'skipped one sample with length {len(data[-1]["msa"][0])}')
            continue
        print(data[-1]['msa'][0])
        try:
            heads = model.test_call(data[-1]['msa'][0])[:, :, 1:, 1:]
        except:
            print('None Error')
            continue
        label = torch.from_numpy(np.array(data[-1]['binary_labels']))
        num_layer, num_head, seqlen, _ = heads.size()
        attentions = heads.view(num_layer * num_head, seqlen, seqlen)
        attentions = apc(symmetrize(attentions))
        attentions = attentions.permute(1, 2, 0)

        proba = net['net'].predict_proba(attentions.reshape(-1, 144).cpu())
        # cor, tot = calculate_contact_precision(data[-1]['name'], torch.from_numpy(proba).to('cuda'), label.to('cuda'), local_range=range_, frac=frac)
        proba = torch.from_numpy(proba).to('cuda').float()
        label = label.to('cuda').float()
        for range_name in range_dic:
            for fra in frac_list:
                cor, tot = calculate_contact_precision(data[-1]['name'], proba.clone(), label.clone(), local_range=range_dic[range_name], local_frac=fra)
                print(cor.item(), tot)
                eval_dic[range_name][fra]['cor'] += cor.item()
                eval_dic[range_name][fra]['tot'] += tot


eval_unsupervised()
print(f'{model.train_sample=}')
print(f'{model.test_sample=}')

print(eval_dic)
for r in range_dic:
    for f in frac_list:
        eval_dic[r][f]['acc'] = eval_dic[r][f]['cor'] / eval_dic[r][f]['tot']


print(eval_dic)
