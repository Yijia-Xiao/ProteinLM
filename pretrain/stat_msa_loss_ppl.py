import os


key = 'validation loss at iteration 10000'

files = os.listdir('./logs/0810/')

stat = []

for f in files:
    with open(os.path.join('logs/0810/', f), 'r') as fd:
        data = fd.readlines()
        for l in data[::-1]:
            if key in l:
                print(l)
                loss = l.split('|')[1].strip().split()[-1]
                ppl = l.split('|')[2].strip().split()[-1]
                stat.append([f, loss, ppl, l])
                break

# print(stat)
for i in stat:
    print(f'| {i[0][4:-30]} | {i[1]} | {i[2]} |')
