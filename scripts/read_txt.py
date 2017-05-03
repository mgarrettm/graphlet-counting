import glob
import sys
import os
import csv

if not len(sys.argv) > 3:
    print 'Error: Usage: python read_txt.py gpu graph [rows]'
    sys.exit()

gpu = sys.argv[1]
graph = sys.argv[2]

# The rest of the argv are considered columns to be fetched.
cols = []
for x in range(3, len(sys.argv)):
    cols.append(sys.argv[x])

# Setup output csv headers.
headers = ['gpu', 'graph', 'block']
for col in cols:
    headers.append(col)

data = []
for path in glob.glob('../output/' + gpu + '___' + graph + '___*.txt'):
    basename = os.path.basename(path)
    name = os.path.splitext(basename)[0]

    # Get run information.
    g, n, b = name.split('___')

    rf = open(path, 'r')
    lines = rf.readlines()
    idata = {'gpu': g, 'graph': n, 'block': b}
    section = False

    x = 0
    while x < len(lines):
        line = lines[x]

        if line.strip().startswith('Timing'):
            x += 2
            break
        x += 1

    while x < len(lines):
        k, v = lines[x].strip().split(':')
        k = k.strip()
        v = v.strip()

        if k in cols:
            idata[k] = v
        x += 1

    data.append(idata)

    with open('out.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        writer.writerows(data)
