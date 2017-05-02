import sys
import re
import time
#print(sys.argv)
#print(len(sys.argv))
headerFlag = 4
if len(sys.argv) != 3: #or len(sys.argv) != 4:
    sys.exit("Usage error\n inputFileName outputFileName [option] header length (4 by default")
else:
    try:
        input = open(sys.argv[1],'r')
    except FileNotFoundError:
        print("Cannot find file")
        sys.exit("Error Opening inputfile")
    try:
        output = open(sys.argv[2],'w+')
    except FileNotFoundError:
        sys.exit("Error Opening inputfile")
for junk in range(headerFlag):
    output.write(input.readline())
nodeNum = int(input.readline())
output.write(str(nodeNum)+"\n")
for nodes in range(nodeNum):
    output.write("|{"+str(nodes+1)+"}|\n")
    input.readline()
edgeNum = int(input.readline())
edgeSet = []
for edge in range(edgeNum):
    if (edge % 1000) == 0:
        percent = (edge/edgeNum)*100
        print(" Working: "+str("%.2f" % percent) +"% done with reading edges",end='\r',flush=True)
    newEdge = input.readline().split()
    for i in range(len(newEdge)):
        newEdge[i] = re.sub('[^0-9]','',newEdge[i])
    newEdge = sorted(newEdge[:2],key = int)
    if newEdge in edgeSet:
        pass
    else:
        edgeSet.append(newEdge)
edgeSet = sorted(edgeSet,key=lambda stuff: (int(stuff[0]),int(stuff[1])))
output.write(str(len(edgeSet))+'\n')
for edge in edgeSet:
    output.write(str(edge[0])+" "+str(edge[1])+" 0 |{}|\n")





















