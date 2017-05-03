import glob
import sys
import os

for path in glob.glob('../output/*.csv'):
    newlines = []

    f = open(path, "r")
    oldlines = f.readlines()
    f.close()

    for line in oldlines:
        if not line.startswith('=='):
            newlines.append(line)
    # print newlines

    basename = os.path.basename(path)
    newpath = '../cleaned-output/' + basename

    wf = open(newpath, 'w')
    wf.writelines(newlines)
    wf.close()
