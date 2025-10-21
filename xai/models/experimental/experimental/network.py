from karateclub import EgoNetSplitter
import networkx as nx
import pandas as pd
import numpy as np 


def rgb2hex(r, g, b):
    return ('#{:02x}{:02x}{:02x}').format(r, g, b)

def int2rgb(RGBint):
    blue =  RGBint & 255
    green = (RGBint >> 8) & 255
    red =   (RGBint >> 16) & 255
    return rgb2hex(red, green, blue)

def cluster_graph(data, source='source', target='target', score='score'):
    data = data.copy()
    node2id = {i:ix for ix,i in enumerate(set(list(data[source]) + list(data[target])))}
    id2node = {v:k for k,v in node2id.items()}
    
    data['source_id'] = [node2id[i] for i in data[source]]
    data['target_id'] = [node2id[i] for i in data[target]]
    
    edges = nx.from_pandas_edgelist(data, 'source_id', 'target_id', edge_attr=True)
    model = EgoNetSplitter(weight=score, resolution=1)
    
    model.fit(edges)
    mbr = model.get_memberships()
    
    clusters = {id2node[k]: v[0] for k,v in mbr.items()}
    
    clst = []
    for ix,i in data.iterrows():
        clg = [clusters[i[source]], clusters[i[target]]]
        clg.sort()
        
        if clusters[i[source]] == clusters[i[target]]:
            clst.append( [clusters[i[source]]] + clg )
        else:
            clst.append([-1] + clg)
            
    return np.array(clst)