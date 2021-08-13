import os
import sys

# '/dataset/f0a0efb9/protein/msa/data/tiny'
data_path=sys.argv[1]
# print(os.listdir(PATH))


files = os.listdir(data_path)

data = []
cnt = 0
for fd in files:
    with open(os.path.join(data_path, fd)) as f:
        msa = f.readlines()
        # sample = []
        sample = ""
        split_add = False
        for l in msa:
            sample += l.strip()
            if not split_add:
                sample += '|'
                split_add = True
        # cnt += 1
        # if cnt == 2:
        #     print(sample)
        #     print(f)
        #     exit(0)
        data.append(sample)

for s in data:
    p_str = ' '.join(s)
    print('{"text": "' + p_str + ' "}')

