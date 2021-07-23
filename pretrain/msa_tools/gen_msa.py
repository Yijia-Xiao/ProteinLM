import os
import sys
import itertools

tokens = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q']


DEPTH = 32
LENGTH = 128
MAX_LENGTH = DEPTH * LENGTH + 1


tot = []
cnt = 0
for p in itertools.permutations(tokens):
    tot.append(p)
    cnt += 1
    if cnt == DEPTH * LENGTH:
        break

# print(len(tot))

repeat = (DEPTH * LENGTH) // len(tokens)
# exit(0)

for p in tot:
    # p_str = ' '.join(p * repeat)
    duplicate = [i for i in p for c in range(repeat)]
    duplicate.insert(repeat, '|')
    p_str = ' '.join(duplicate)
    print('{"text": "' + p_str + '"}')
exit(0)

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

    # contact = data[0] + "|"
    concat = ''.join(sample)
    print('{"text": "' + ' '.join(concat) + '"}')
