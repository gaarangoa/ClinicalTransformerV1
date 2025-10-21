from tqdm.auto import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def inner_cosine_distance(patient_embeddings, labels=[]):
    pex = []
    for pix in tqdm(range( len(patient_embeddings) )):
        emb = pd.concat(patient_embeddings[pix]['embeddings'], axis=1).T

        emx = pd.DataFrame(cosine_similarity(emb), index=emb.index, columns=emb.index)
        emx['id'] = emx.index

        emx = emx.melt(id_vars=['id'])

        emx['id'] = [i.split('-')[1] for i in emx.id]
        emx['variable'] = [i.split('-')[1] for i in emx.variable]

        emx = emx.groupby(['id', 'variable']).median().reset_index()
        emx.columns = ['source', 'target', 'score']

        emx['score'] = (emx.score - emx.score.min()) / (emx.score.max() - emx.score.min())

        emx['id'] = pix
        if len(labels) == 0:
            emx['label'] = patient_embeddings[pix]['label']
        else:
            emx['label'] = labels[pix]

        pex.append(emx)

    return pd.concat(pex)

