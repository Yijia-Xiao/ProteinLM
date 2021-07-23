import os
import sys
DEPTH = 128
LENGTH = 1024
MAX_LENGTH = DEPTH * LENGTH + 1
# folderpath = '/dataset/f0a0efb9/protein/msa/protein-msa/pretrain/msa/a2m'
folderpath = sys.argv[1]

msa = []
files = os.listdir(folderpath)
template = '{"text": "{}"}'

for f in files:
    with open(os.path.join(folderpath, f), 'r') as fd:
        raw = fd.readlines()
    data = [line.strip() for line in raw]
    sample = []
    msa_depth = min(DEPTH, len(data))
    for seq_ in data[: msa_depth]:
        seq = seq_.strip()
        seq = seq[: min(LENGTH, len(seq))]
        # sample.append(' '.join(seq.ljust(LENGTH - 1, '~')))
        sample.append(seq.ljust(LENGTH, '~'))
    for i in range(DEPTH - msa_depth):
        sample.append('~' * LENGTH)

    concat = ''.join(sample)
    print('{"text": "' + ' '.join(concat) + '"}')
