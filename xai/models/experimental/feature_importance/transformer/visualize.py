import networkx as nx
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_feature_network(cssx, ax=None, edge_alpha=0.001, q=0.75, nodePos=False, draw=True, nodeFactor=100, less_than=False):
    
    node_sizes_df = cssx[cssx.source == cssx.target].reset_index(drop=True)

    node_sizes_df['score'] = (node_sizes_df['score'] - np.min(node_sizes_df['score'])) / (np.max(node_sizes_df['score']) - np.min(node_sizes_df['score']))

    thr = np.quantile(cssx[cssx.source != cssx.target].score, q)
    cssx = cssx[(cssx.source != cssx.target) & (cssx.score >= thr)].reset_index(drop=True)
    cssx['score'] = (cssx['score'] - np.min(cssx['score'])) / (np.max(cssx['score']) - np.min(cssx['score']))

    if not ax:
        f=plt.figure(constrained_layout = True, figsize=(15, 5) )
        gs = f.add_gridspec(1, 3)
        ax = f.add_subplot(gs[0, 0])

    H = nx.from_pandas_edgelist(cssx, edge_attr='score')
    if not nodePos:
        nodePos = nx.shell_layout(H)

    ax.set_xlim([
        np.min([j[0] for i,j in nodePos.items()]) - 0.5,
        np.max([j[0] for i,j in nodePos.items()]) + 0.5
    ]);
    
    ax.set_ylim([
        np.min([j[1] for i,j in nodePos.items()]) - 0.5,
        np.max([j[1] for i,j in nodePos.items()]) + 0.5
    ]);

    node_color_map_values = np.array([1 for i in H.nodes()])
    norm = matplotlib.colors.Normalize(vmin=np.min(node_color_map_values), vmax=np.max(node_color_map_values), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Wistia)

    node_size = {i.source: i.score for ix,i in node_sizes_df.iterrows()}
    node_size_map = [nodeFactor*(node_size[i] + 0) for i in H.nodes()]
    
    edges = nx.get_edge_attributes(H,'score')

    node_color_map = [ 'lightgreen' for i in node_color_map_values]

    label_dict = {i:i.replace('molecular_', '') for i in H.nodes()}

    if draw:
        nx.draw(H, nodePos, node_color=node_color_map, node_size=node_size_map, alpha=0.91, width=0.0, ax=ax, with_labels=True, font_size=12, labels=label_dict, font_weight="bold")
        nx.draw_networkx_edges(H, nodePos, edgelist=list(edges.keys()), alpha=0.8, width=(edge_alpha*np.array(list(edges.values()))), ax=ax, edge_color='lightblue'); 
    
    return nodePos

def plot_feature_importance_cosine(pexi, ax=[]):
    if ax:
        sns.boxplot(data=pexi[pexi.source == pexi.target], x='source', y='score',
            order=pexi[pexi.source == pexi.target].groupby('source').std().reset_index().sort_values('score', ascending=False).source,
            ax=ax
        )
    else:
        sns.boxplot(data=pexi[pexi.source == pexi.target], x='source', y='score',
            order=pexi[pexi.source == pexi.target].groupby('source').std().reset_index().sort_values('score', ascending=False).source,
        )