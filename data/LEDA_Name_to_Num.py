import sys
import re
import time
#print(sys.argv)
#print(len(sys.argv))
headerFlag = 4
if len(sys.argv) != 3: #or len(sys.argv) != 4:
    sys.exit("Usage error\n inputFileName outputFileName [option] header length (4 by default")
else:
    #Open input file
    try:
        input = open(sys.argv[1],'r')
    except FileNotFoundError:
        print("Cannot find file")
        sys.exit("Error Opening inputfile")
    #Create output file
    try:
        output = open(sys.argv[2],'w+')
    except FileNotFoundError:
        sys.exit("Error Opening inputfile")
"""
    if len(sys.argv) == 4:
    try:
        headerFlag = int(sys.argv[4])
    except ValueError:
        sys.exit("Header length is not an int")
"""

for junk in range(headerFlag):
    output.write(input.readline())

#need to do a check
nodeNum = int(input.readline())
output.write(str(nodeNum)+"\n")
#starts with 1 to N
for nodes in range(nodeNum):
    output.write("|{"+str(nodes+1)+"}|\n")
    input.readline()
#need to check
edgeNum = int(input.readline())
edgeSet = []
print("entering put all the edges into list")
for edge in range(edgeNum):
    if (edge % 1000) == 0:
        percent = edge/edgeNum
        print(" Working: "+str("%.2f" % percent) +"% done with reading edges",end='\r',flush=True)
    newEdge = input.readline().split()
    for i in range(len(newEdge)):
        newEdge[i] = re.sub('[^0-9]','',newEdge[i])
    newEdge = sorted(newEdge[:2],key = int)
    if newEdge in edgeSet:
        pass
        #print("already there!!")
    else:
        edgeSet.append(newEdge)
print("exiting and going to sort")
edgeSet = sorted(edgeSet,key=lambda stuff: (int(stuff[0]),int(stuff[1])))
output.write(str(len(edgeSet))+'\n')
for edge in edgeSet:
    output.write(str(edge[0])+" "+str(edge[1])+" 0 |{}|\n")





















