"""
cluster.py
"""
import sys
import time
import pickle
import networkx as nx
import matplotlib.pyplot as plt

def create_graph(friendIds, followerIds):
    graph = nx.Graph()
    for k,v in friendIds.items():
        for i in v:
            graph.add_edges_from([(k,str(i))])
    for k,v in followerIds.items():
        for i in v:
            graph.add_edges_from([(k,str(i))])
    return graph

def partition_girvan_newman(graph, max_depth):
    copiedGraph = graph.copy()
    betweenness = nx.edge_betweenness_centrality(copiedGraph)
    reversedBetweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    components = [c for c in nx.connected_component_subgraphs(copiedGraph)]
    while len(components) == 1:
        copiedGraph.remove_edge(*reversedBetweenness[0][0])
        del reversedBetweenness[0]
        components = [c for c in nx.connected_component_subgraphs(copiedGraph)]

    return components

def draw_graph(graph, screenNames, filename):
    user_list = {}
    for screenName in screenNames:
        if type(screenName) == str:
            user_list[screenName] = screenName
        else:
            user_list[screenName] = ''
    plt.figure(figsize=(50, 50), dpi=None, facecolor=None, edgecolor='black', linewidth=1.0, frameon=None,
               subplotpars=None, tight_layout=None)
    nx.draw_networkx(graph,node_colors='r', labels=user_list, font_size=30, width=1, node_size=80)
    plt.axis('off')
    plt.savefig(filename)
    plt.show(block=False)

def main():
    print("*********Clustering Phase*********")
    collectedDataFileName = 'collected_data.p'
    clusterFileName = 'clustered_data.p'
    data = pickle.load(open(collectedDataFileName,'rb'))
    filename = 'network_graph.png'
    screenNames = data['ScreenNames']
    friendIds = data['Friend_Ids']
    followerIds = data['Follower_Ids']
    graph = create_graph(friendIds,followerIds)
    clusters = partition_girvan_newman(graph, 3)
    for i in range(len(clusters)):
        print("cluster %d has %d nodes " %((i+1), clusters[i].order()))
        print("cluster %d nodes:" %(i+1))
        print(clusters[i].nodes())
    pickle.dump(clusters,open(clusterFileName,'wb'))
    draw_graph(graph, screenNames,filename)

if __name__ == '__main__':
    main()