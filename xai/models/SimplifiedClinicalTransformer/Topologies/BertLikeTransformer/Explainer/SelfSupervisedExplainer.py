import numpy as np 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

def extract_patient_embeddings(data, sample_id, o, t, **kwargs):
    '''
    Parse output from the transformer and get the outputs, features, 
    patient ids and iterations as a giant table. This is useful for downstreatm tasks
    '''
    patients = data[sample_id]
    predicted_labels = kwargs.get('labels', False)
    true_labels = kwargs.get('true_labels', False)

    outputs = []
    labels = []
    true_labels_out = []
    features = []
    patient_ids = []
    iterations = []
    risks = []
    for iteration in range(o.shape[0]):
        for patient_ix, patient in enumerate(patients):

            label = predicted_labels[iteration, patient_ix, :, :]
            true_label = true_labels[iteration, patient_ix, :, :]

            ô = o[iteration, patient_ix, :, :]
            t̂ = t[iteration, patient_ix, :]

            no_pad_out = ô[ (t̂ != '<pad>')]
            no_pad_features = t̂[ (t̂ != '<pad>')]
            no_pad_label = label[(t̂ != '<pad>')]
            no_pad_true_label = true_label[(t̂ != '<pad>')]

            outputs.append(no_pad_out)
            features.append(no_pad_features)
            patient_ids.append([patient]*len(no_pad_features))
            iterations.append([iteration]*len(no_pad_features))
            labels.append(no_pad_label)
            true_labels_out.append(no_pad_true_label)

    outputs = np.concatenate(outputs, axis=0)
    features = np.concatenate(features)
    patient_ids = np.concatenate(patient_ids)
    iterations = np.concatenate(iterations)
    labels = np.concatenate(labels)
    true_labels_out = np.concatenate(true_labels_out)
    
    return outputs, features, patient_ids, iterations, labels, true_labels_out

def inner_cosine_scores(patient, features, outputs, iterations, patient_ids):
    '''
    From the embeddings, compute the cosine distances and score feature interactions
    '''
    
    patient_features = features[patient_ids == patient]
    patient_outputs = outputs[patient_ids == patient]
    patient_iterations = iterations[patient_ids == patient]

    # Compute cosine similarity
    cos = cosine_similarity(patient_outputs)

    # store it in a data frame 
    emx = pd.DataFrame(cos, index=patient_features, columns=patient_features)
    emx['id'] = emx.index

    # Only take the upper triangular matrix
    emx = emx.where(np.triu(np.ones(emx.shape)).astype(bool))

    # Transform the matrix to a vector
    emx = emx.melt(id_vars=['id'])
    emx = emx.dropna()
    # emx['value']  = emx.value.abs()

    # Average variable cosine similarity by their iterations (e.g, if 10 iterations returns the average over the 10 random feature sampling iterations)
    emx = emx.groupby(['id', 'variable']).mean().reset_index()
    emx.columns = ['source', 'target', 'score']

    # Normalize feature interactions (cosine distance)
    # emx['score'] = (emx.score - emx.score.min()) / (emx.score.max() - emx.score.min())
    emx['id_'] = patient

    return emx

def compute_attention_scores(data, features, outputs, iterations, patient_ids, sample_id='SAMPLE_ID'):
    emxso = []
    for pix, patient in tqdm(data.iterrows(), total=data.shape[0]):
        emx = inner_cosine_scores(patient[sample_id], features, outputs, iterations, patient_ids)
        # emx['pred_prob'] = patient.pred_prob
        # emx['pred_label'] = patient.pred_label

        emxso.append(emx)
    
    emxs = pd.concat(emxso)
    return emxs

def selfsupervision_attention_scores(data, evaluator, iterations=2, sample_id='SAMPLE_ID', batch_size=10000, **kwargs):
    data = data.copy()
    
    return_attentions = kwargs.get('return_attentions', False)

    # The iteration in this function is how many times a single patient is subsampled (feature space). 
    β, w, o, t, true_labels = evaluator.predict(
        data, iterations=iterations, normalize=True, batch_size=batch_size
    )
    
    # Compute risk score    
    outputs, features, patient_ids, iters, labels, true_labels_out = extract_patient_embeddings(data, sample_id, o, t, labels=β, true_labels=true_labels)
    
    # Sort by feature names and do it for all the other vectors. Thus, we will have the same feature order and edges (interactions).
    sf = np.argsort(features)
    outputs  = outputs[sf]
    features = features[sf]
    patient_ids = patient_ids[sf]
    iters = iters[sf]
    labels = labels[sf]
    true_labels_out = true_labels_out[sf]

    # The iterations in this parameter refers to the iteration vectors (repetitions of the same patient)
    attention_scores = []
    if return_attentions:
        attention_scores = compute_attention_scores(data, features, outputs, iters, patient_ids, sample_id=sample_id)
    
    return labels, outputs, features, patient_ids, iters, attention_scores, true_labels_out
