
import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt

arr = ['ccr7', 'cd27','cd28','cd45ra','cd57','cd279','KLRG1']

def distance(A1,A2): 
  return np.sum((np.ravel(A1) - np.ravel(A2))**2)

import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(graph, graph1, graph2, weight1, weight2):

    # extract nodes from graph
    nodes = set([n1 for n1, n2 in graph] + [n2 for n1, n2 in graph])

    # create networkx graph
    G=nx.Graph()

    # add nodes
    for node in nodes:
        G.add_node(node)

    # add edges
    i = 0 
    for edge in graph1:
        G.add_edge(edge[0], edge[1], color = 'r', weight = weight1[i])
        i += 1

    j = 0
    for edge in graph2:
        G.add_edge(edge[0], edge[1], color = 'b', weight = weight2[j])
        j += 1

    # draw graph
    #pos = nx.shell_layout(G)
    pos = nx.circular_layout(G)

    edges = G.edges()
    colors = [G[u][v]['color'] for u,v in edges]
    weights = [G[u][v]['weight'] for u,v in edges]
    plt.figure(figsize=(9,6))
    nx.draw(G, pos, node_color='none', edge_color=colors, width=weights, with_labels=False, font_size= 20, node_size = 10)
    for node, (x, y) in pos.items():
      plt.text(x, y, node, fontsize=30, ha='center', va='center')

    # show graph
    plt.savefig('lean_graph.pdf')
    #plt.show()
# draw example
draw_graph(graph, graph1, graph2, weight1, weight2)

"""# Ising Model for CD4 and CD8"""

def A_thres(A, thres):
  A [np.abs(A)< thres] = 0
  X = np.around(A, 2)
  #print(X)
  XX = np.triu(X)
  XXX = []
  for i in range(XX.shape[0]):
    for j in range(i+1, XX.shape[0]):
      XXX.append(XX[i][j]) 
  #print(XXX) 

  XX = np.tril(X).T
  XXX2 = []
  for i in range(XX.shape[0]):
    for j in range(i+1, XX.shape[0]):
      XXX2.append(XX[i][j]) 
  #print(XXX2) 

  res = []
  for i in range(len(XXX)):
    if XXX[i] >= 0 and XXX2[i] >= 0:
      res.append(max(XXX[i], XXX2[i]))
    elif XXX[i] <= 0 and XXX2[i] <= 0:
      res.append(min(XXX[i], XXX2[i]))
    else:
      print(bla)
  #print(res)
  return res

A_lean_CD4 = np.load('Carma_lean/A_lean_mean_CD4.npy')
A_ob_CD4 = np.load('Carma_ob/A_ob_mean_CD4.npy')
A_obd_CD4 = np.load('Carma_obd/A_obd_mean_CD4.npy')

print(distance(A_lean_CD4, A_ob_CD4), distance(A_lean_CD4, A_obd_CD4), distance(A_obd_CD4, A_ob_CD4))


t = 0.115
res = A_thres(A_lean_CD4, thres = t)
print('lean_CD4:', res)

res = A_thres(A_ob_CD4, thres = t)
print('ob_CD4:', res)

res = A_thres(A_obd_CD4, thres = t)
print('obd_CD4:', res)

A_lean_CD8 = np.load('Carma_lean/A_lean_mean_CD8.npy')
A_ob_CD8 = np.load('Carma_ob/A_ob_mean_CD8.npy')
A_obd_CD8 = np.load('Carma_obd/A_obd_mean_CD8.npy')

print(distance(A_lean_CD8, A_ob_CD8), distance(A_lean_CD8, A_obd_CD8), distance(A_obd_CD8, A_ob_CD8))

t = 0.121
res = A_thres(A_lean_CD8, thres = t)
print('lean_CD8:', res)

res = A_thres(A_ob_CD8, thres = t)
print('ob_CD8:', res)

res = A_thres(A_obd_CD8, thres = t)
print('obd_CD8:', res)

"""# Ising Model"""

A_lean = np.load('Carma_lean/A_lean_mean.npy')
A_ob = np.load('Carma_ob/A_ob_mean.npy')
A_obd = np.load('Carma_obd/A_obd_mean.npy')

print(distance(A_lean, A_ob), distance(A_lean, A_obd), distance(A_obd, A_ob))

plt.imshow((A_lean - A_ob)**2)

(A_lean - A_ob)

(A_lean - A_ob)**2

thres = 0.15
A_obd [np.abs(A_obd )< thres] = 0
X = np.around(A_obd, 2)
X



plt.imshow(A_lean)
plt.savefig('A_lean.pdf')

plt.imshow(A_ob)
plt.savefig('A_ob.pdf')

plt.imshow(A_obd)
plt.savefig('A_obd.pdf')

X2 = A_lean
thres = 0.15


X2[np.abs(X2)< thres] = 0


graph = []
graph1 = []
graph2 = []
weight1 = []
weight2 = []
for i, el in enumerate(arr):
  for j, el2 in enumerate(arr):
    if X2[i][j] != 0:
      graph.append((el,el2))
      if X2[i][j] > 0:
        graph1.append((el,el2))
        weight1.append( X2[i][j]*5 )
      else:
        graph2.append((el,el2))
        weight2.append(-X2[i][j]*5)

# draw example
draw_graph(graph, graph1, graph2, weight1, weight2)

graph1

X2 = A_ob


X2[np.abs(X2)< thres] = 0


graph = []
graph1 = []
graph2 = []
weight1 = []
weight2 = []
for i, el in enumerate(arr):
  for j, el2 in enumerate(arr):
    if X2[i][j] != 0:
      graph.append((el,el2))
      if X2[i][j] > 0:
        graph1.append((el,el2))
        weight1.append( X2[i][j] *5 )
      else:
        graph2.append((el,el2))
        weight2.append(-X2[i][j]*5)

# draw example
draw_graph(graph, graph1, graph2, weight1, weight2)

X2 = A_obd


X2[np.abs(X2)< thres] = 0


graph = []
graph1 = []
graph2 = []
weight1 = []
weight2 = []
for i, el in enumerate(arr):
  for j, el2 in enumerate(arr):
    if X2[i][j] != 0:
      graph.append((el,el2))
      if X2[i][j] > 0:
        graph1.append((el,el2))
        weight1.append( X2[i][j] *5 )
      else:
        graph2.append((el,el2))
        weight2.append(-X2[i][j] * 5)

# draw example
draw_graph(graph, graph1, graph2, weight1, weight2)

X2

"""# Glasso Model"""

A_lean = np.load('Carma_lean/glasso_lean_mean.npy')
np.fill_diagonal(A_lean, 0)
A_ob = np.load('Carma_ob/A_ob_glasso.npy')
np.fill_diagonal(A_ob, 0)
A_obd = np.load('Carma_obd/A_obd_glasso.npy')
np.fill_diagonal(A_obd, 0)

print(distance(A_lean, A_ob), distance(A_lean, A_obd), distance(A_obd, A_ob))



thres = 1
A_obd [np.abs(A_obd )< thres] = 0
X = np.around(-A_obd, 2)
X

print(np.sqrt(distance(A_lean, A_ob)), np.sqrt(distance(A_lean, A_obd)), np.sqrt(distance(A_obd, A_ob)))

A_lean[np.abs(A_lean) < 1] = 0
plt.imshow(A_lean)
plt.savefig('glasso_lean.pdf')

sorted(np.ravel(A_lean))

plt.imshow(A_ob)
plt.savefig('glasso_ob.pdf')

plt.imshow(A_obd)
plt.savefig('glasso_obd.pdf')

"""# Correlation Model"""

A_lean = np.load('Carma_lean/corr_lean_mean.npy')
A_ob = np.load('Carma_ob/A_ob_corr.npy')
A_obd = np.load('Carma_obd/A_obd_corr.npy')

print(distance(A_lean, A_ob), distance(A_lean, A_obd), distance(A_obd, A_ob))

thres = 0.15
A_lean [np.abs(A_lean )< thres] = 0
X = np.around(A_lean, 2)
X

plt.imshow(A_lean)
plt.savefig('corr_lean.pdf')

plt.imshow(np.linalg.inv(A_lean))

plt.imshow(A_ob)
plt.savefig('corr_ob.pdf')

plt.imshow(A_obd)
plt.savefig('corr_obd.pdf')

A_lean = np.load('Carma_lean/A_lean_miic.npy')
A_ob = np.load('Carma_ob/A_ob_miic.npy')
A_obd = np.load('Carma_obd/A_obd_miic.npy')

print(distance(A_lean, A_ob), distance(A_lean, A_obd), distance(A_obd, A_ob))

plt.imshow(A_lean.reshape(1,-1))
plt.savefig('miic_lean.pdf')

plt.imshow(A_ob.reshape(1,-1))
plt.savefig('miic_ob.pdf')

plt.imshow(A_obd.reshape(1,-1))
plt.savefig('miic_obd.pdf')

100*distance(A_obd, A_ob)

