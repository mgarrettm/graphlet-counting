import glob
import sys
import os
import csv

if not len(sys.argv) > 3:
    print 'Error: Usage: python read_csv.py gpu graph [colums]'
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
for path in glob.glob('../cleaned-output/' + gpu + '___' + graph + '___*.csv'):
    basename = os.path.basename(path)
    name = os.path.splitext(basename)[0]

    # Get run information.
    g, n, b = name.split('___')

    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)

        x = 0
        rd = {'gpu': g, 'graph': n, 'block': b}
        for row in reader:
            keys = row.keys()

            if x == 1:
                for col in cols:
                    if col in keys:
                        rd[col] = row[col]

            x += 1
        data.append(rd)

    with open('out.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        writer.writerows(data)
