import pandas as pd
import numpy as np 
from gprofiler import GProfiler

def gene_pathways_finder(feature_list, keyword, pval=1e-5):
    '''
    From a list of genes generate 
    a matrix with all the pathways 
    associated to the list of genes
    
    Input
    ----------
    list of genes: 
    keyword:    The dataframe should have a prefix to the gene name. 
                For instance moleculr_TP53. The prefix is "molecular_"

    Output:   
    ----------------------------------------------------------------
    Returns a dataframe where the rows are pathways / GO terms and the 
    columns are the genes in the list. 

    '''
    genes = [i.replace(keyword, '') for i in feature_list]
    
    gp = GProfiler(return_dataframe=True)
    gr = gp.profile(
        organism='hsapiens',
        query=genes,
        all_results=False, 
        no_evidences=False
    )

    mapping = []
    for px, p in gr.iterrows():
        for gene in p.intersections:
            if p.p_value < pval:
                mapping.append({'gene': gene, 'ontology': p.native})


    mapping = pd.DataFrame(mapping)
    mapping['Counter'] = 1

    mapping = mapping.pivot(index='gene', columns='ontology', values='Counter').reset_index().fillna(0)
    mapping.index = mapping.gene

    mapping = mapping.iloc[:, 1:].T.reset_index()

    mapping.columns = keyword + mapping.columns 
    
    return mapping