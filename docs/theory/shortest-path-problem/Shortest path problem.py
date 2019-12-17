#!/usr/bin/env python
# coding: utf-8

# In[24]:


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

#Graph Adjacency Matrix

Adj=np.asarray([[0, 156, 0, 0, 246, 0, 184, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 462, 0, 0, 171, 0, 157, 0, 363], 
[156, 0, 323, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 323, 0, 151, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 151, 0, 0, 545, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[246, 0, 0, 0, 0, 174, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 545, 174, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[184, 0, 0, 0, 0, 0, 0, 83, 224, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 192, 0, 0, 0], 
[0, 0, 0, 0, 100, 0, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 224, 0, 0, 209, 0, 0, 0, 0, 217, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 209, 0, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 116, 0, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 180, 0, 157, 251, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 157, 0, 342, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 251, 342, 0, 111, 208, 0, 0, 0, 0, 0, 382, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 217, 0, 0, 0, 0, 111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 208, 0, 0, 335, 462, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 335, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[462, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 462, 0, 0, 212, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 212, 0, 135, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 135, 0, 174, 0, 0, 0, 0], 
[171, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 174, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 192, 0, 0, 0, 0, 0, 0, 382, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 171, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 171, 0, 0], 
[363, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])



#Plot the directed graph
G = nx.DiGraph()

N = Adj.shape[0]
for i in range(N):
    G.add_node(i)
  
for i in range(N):
    for j in range(N):
        if Adj[i,j] > 0:
            G.add_edges_from([(i, j)], weight=Adj[i,j])

print("Graph plotting:")

pos=nx.spring_layout(G) 
edge_labels=dict([((u,v,),d['weight'])
                 for u,v,d in G.edges(data=True)]) 
nx.draw_networkx(G,pos,edge_labels=edge_labels, node_size = 1000, node_color = 'y') 
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.axis('off')
plt.show()

#Dijkstra algorithm
distance = np.zeros(N) 
visited = np.ones(N) 
origin = 0
goal = 4

visited[origin] = 0

pred = np.zeros(N)
pred[origin] = origin

for j in range(N):
  if Adj[origin,j] == 0 and origin != j: 
    distance[j] = 10e10
    pred[j] = -1
  else:
    distance[j] = Adj[origin,j]
    pred[j] = origin

array = []
array.append(0)
counter = 1

while(np.sum(visited) > 0): 
  temp = np.copy(distance) 
  temp[visited == 0] = 10e10
  vmin = np.argmin(temp)

  visited[vmin] = 0
  for j in range(N):
    counter+=1
    if Adj[vmin,j] > 0 and distance[j] > distance[vmin] + Adj[vmin,j]: 
      distance[j] = distance[vmin]+Adj[vmin,j]
      array.append(counter)
      pred[j] = vmin
        
#print("counter:", array)
pred = pred.astype(int) #Minimum distance path from origin node to the others
#print("Pred")
#print(pred)
#Plot path

dist_list = []
prev_list = []

dist_list.append(distance[goal])
prev_list.append(goal)

previous = pred[goal]
path = [(previous, goal),(goal, previous)]
print("The minimum distance path from "+str(origin)+" to "+str(goal)+" is: "+str(goal)+" <-- "+str(previous), end="")



while(previous != origin):
  path.append((previous, pred[previous]))
  path.append((pred[previous], previous))
  dist_list.append(distance[previous])
  prev_list.append(previous)
  previous = pred[previous]
  print(" <-- "+str(previous), end="")
dist_list.append(distance[previous])

#print(dist_list)
#print(prev_list)

edge_colors = ['black' if not edge in path else 'red' for edge in G.edges()]

pos=nx.spring_layout(G)
edge_labels=dict([((u,v,),d['weight'])
                 for u,v,d in G.edges(data=True)])
nx.draw_networkx(G,pos,edge_labels=edge_labels, node_size = 1000, node_color = 'y', edge_color=edge_colors)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.axis('off')
plt.show()


dist_list.reverse()
dist_list = [dist_list[-1] - x for x in dist_list]

plt.plot(array[:len(dist_list)], dist_list, label='Dijkstra')
plt.legend()
plt.xlabel('Iterations') 

plt.ylabel('Distance')
plt.show()


# In[25]:


from matplotlib import pyplot as plt
import numpy as np
''' Part of Cosmos by OpenGenus Foundation '''
INF = 1000000000

def floyd_warshall(vertex, adjacency_matrix,a,b):

    
    # calculating all pair shortest path
    counter = 0
    arr_value = []
    arr_counter = []
    prev_value= 0
    
    
    
    for k in range(0, vertex):
        for i in range(0, vertex):
            for j in range(0, vertex):
                # relax the distance from i to j by allowing vertex k as intermediate vertex
                # consider which one is better, going through vertex k or the previous value
                counter += 1
                adjacency_matrix[i][j] = min(adjacency_matrix[i][j], adjacency_matrix[i][k] + adjacency_matrix[k][j])
                if(adjacency_matrix[a][b] != prev_value ):
                    prev_value = adjacency_matrix[i][j]
                    if(prev_value != INF):
                        arr_counter.append(counter)
                        arr_value.append(prev_value)
                        #print(prev_value)
                        #print(counter)
                
  



    return arr_counter, arr_value, adjacency_matrix[a][b]


for i in range(len(Adj)):
    for j in range(len(Adj)):
        if(Adj[i][j] == 0 and i != j):
            Adj[i][j] = INF

x, y, max_value = floyd_warshall(len(Adj), Adj, 0, 4)

y = [max_value - i for i in y]

plt.plot(x, y, label='Floyd–Warshall')
plt.legend()
plt.xlabel('Iterations') 
plt.ylabel('Distance')


# In[26]:


plt.plot(x, y, label='Floyd–Warshall')
plt.plot(array[:len(dist_list)], dist_list, label='Dijkstra')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Distance')


# In[ ]:




