import sys

edges = {}
seen = {}
offsets = {}
max_vertex = 0
s = open(sys.argv[2], "w+")
with open(sys.argv[1]) as f:
    for line in f:
        arr = line.split("\t")
        u = int(arr[0])
        v = int(arr[1])
        seen[u] = True
        seen[v] = True
        if u not in edges:
            edges[u] = {}
        if v not in edges:
            edges[v] = {}
        edges[u][v] = True
        edges[v][u] = True
        if max_vertex < u:
            max_vertex = u
        if max_vertex < v:
            max_vertex = v

offset = 0
for u in range(max_vertex + 1):
    if u in seen:
        offsets[u] = offset
    else:
        offset += 1

for u in sorted(edges.keys()):
    for v in sorted(edges[u].keys()):
        if u != v and u in edges and v in edges[u] and u < v:
            s.write(str(u - offsets[u] + 1) + "\t" + str(v - offsets[v] + 1) + "\n")
