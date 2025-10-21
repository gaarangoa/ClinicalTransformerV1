import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import matplotlib
import numpy as np 

def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % (rgb[0], 0, rgb[2])
    
def draw_network(data, score='', ax=None, nodePos=[], **kargs):

    try:    
        font_size = kargs['font_size']
    except:
        font_size = 12
    
    try:
        edge_alpha = kargs['edge_alpha']
    except:
        edge_alpha = 1

    try:
        with_labels = kargs['with_labels']
    except:
        with_labels = True
    
    try:
        replace_names = kargs['replace_names']
    except:
        replace_names = {}

    try:
        layout = kargs['layout']
    except:
        layout = 'circular'
    
    for k,v in replace_names.items():
        data['source'] = data['source'].replace({k:v})
        data['target'] = data['target'].replace({k:v})
    
    if not ax:
        f=plt.figure(constrained_layout = True, figsize=(15, 5) )
        gs = f.add_gridspec(1, 3)
        ax = f.add_subplot(gs[0, 0])

    H = nx.from_pandas_edgelist(data, edge_attr=True)
    if not nodePos:
        if layout == 'circular':
            nodePos = nx.circular_layout(H)
        if layout == 'spring':
            nodePos = nx.spring_layout(H, iterations=100, scale=4, k=2)
        if layout == 'shell': 
            nodePos = nx.shell_layout(H, scale=4)

    try:
        edge_color = nx.get_edge_attributes(H,'color').values()
        edge_color[0]
    except:
        edge_color = 'red'

    ax.set_xlim([
        np.min([j[0] for i,j in nodePos.items()]) - 0.5,
        np.max([j[0] for i,j in nodePos.items()]) + 0.5
    ]);
    
    ax.set_ylim([
        np.min([j[1] for i,j in nodePos.items()]) - 0.5,
        np.max([j[1] for i,j in nodePos.items()]) + 0.5
    ]);

    edges = H.edges()
    weights = nx.get_edge_attributes(H,score)
    
    colors = [rgb_to_hex(tuple(np.repeat(int(255 * (1-weights[edge])),255))) for edge in edges]
    
    node_color_map_values = np.array([1 for i in H.nodes()])
    norm = matplotlib.colors.Normalize(vmin=np.min(node_color_map_values), vmax=np.max(node_color_map_values), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Wistia)

    node_size_map = [500 for i in H.nodes()]
    
    edges = nx.get_edge_attributes(H, score)
    node_color_map = [ 'white' for i in node_color_map_values]

    label_dict = {i:i.replace('molecular_', '') for i in H.nodes()}

    nx.draw(
        H, nodePos, node_color=node_color_map, 
        node_size=node_size_map, alpha=0.7, 
        ax=ax, with_labels=with_labels, 
        font_size=font_size, labels=label_dict, 
        font_weight="bold",
        font_color='black',
        edge_color=edge_color,
        width=(edge_alpha*np.array(list(edges.values())))
    )