import os
import sys
import tqdm

data_path=sys.argv[1]
MAX_DEPTH=int(sys.argv[2])
MAX_LEN=int(sys.argv[3])

files = os.listdir(data_path)

data = []
cnt = 0
for fd in tqdm.tqdm(files):
    with open(os.path.join(data_path, fd)) as f:
        msa = f.readlines()
        if len(msa) < MAX_DEPTH or len(msa[0]) < MAX_LEN:
            continue
        sample = ""
        split_add = False
        for l in msa:
            sample += l.strip()
            if not split_add:
                sample += '|'
                split_add = True
        data.append(sample)

for s in data:
    p_str = ' '.join(s)
    print('{"text": "' + p_str + ' "}')

