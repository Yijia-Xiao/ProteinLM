import argparse
from tqdm import tqdm
import lmdb
import pickle as pkl
import numpy as np
from Bio import SeqIO


# python fasta_to_lmdb.py input.fasta output.lmdb
parser = argparse.ArgumentParser(description='Convert a fasta file into an lmdb file. e.g. python fasta_to_lmdb.py input.fasta output.lmdb')

parser.add_argument('fastafile', type=str, help='The fasta file to convert')
parser.add_argument('lmdbfile', type=str, help='The lmdb file to output')

args = parser.parse_args()
ids = list()
seqs = list()

lmdbfile = args.lmdbfile
if not lmdbfile.endswith('.lmdb'):
    lmdbfile += '.lmdb'

data = list()
for record in SeqIO.parse(args.fastafile, "fasta"):
    data.append((record.id, str(record.seq)))


env = lmdb.open(str(lmdbfile), map_size=50e9)
with env.begin(write=True) as txn:
    # count total number of samples
    num_examples = 0
    for i, example in enumerate(tqdm(data)):
        item = dict()
        item['id'] = example[0]
        length = len(example[1])
        item['protein_length'] = length
        item['seq_len'] = length
        item['ss3'] = np.array([0] * length)
        primary = ""
        for c in example[1]:
            primary += c + ' '
        item['primary'] = primary[:-1]

        id_ = str(i).encode()
        txn.put(id_, pkl.dumps(item))
        num_examples += 1
    txn.put(b'num_examples', pkl.dumps(num_examples))
