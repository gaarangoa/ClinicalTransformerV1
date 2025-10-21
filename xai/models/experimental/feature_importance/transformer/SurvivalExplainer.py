from xai.models.SimplifiedClinicalTransformer.embeddings import extractor
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import ttest_ind_from_stats

from xai.models.SimplifiedClinicalTransformer.SurvivalEstimator import SurvivalEstimator
from xai.models.SimplifiedClinicalTransformer.performance import load_model
from xai.metrics.survival import sigmoid_concordance
from xai.models.SimplifiedClinicalTransformer.embeddings import extractor
from sklearn.metrics.pairwise import cosine_similarity

np.random.seed(0)
cmap = matplotlib.cm.get_cmap("Spectral")

def extract_patient_embeddings(data, sample_id, o, t, r, iterations_=50 ):
    '''
    Parse output from the transformer and get the outputs, features, patient ids and iterations. 
    '''
    patients = data[sample_id]

    outputs = []
    features = []
    patient_ids = []
    iterations = []
    # risks = []
    for iteration in range(iterations_):
        for patient_ix, patient in enumerate(patients):

            ô = o[iteration, patient_ix, :, :]
            t̂ = t[iteration, patient_ix, :]

            ri = r[iteration, patient_ix, :]

            no_pad_out = ô[ (t̂ != '<pad>') & (t̂ != '<mask>') ]
            no_pad_features = t̂[ (t̂ != '<pad>') & (t̂ != '<mask>') ]
            # no_pad_risk = ri[ (t̂ != '<pad>') & (t̂ != '<mask>') ]

            outputs.append(no_pad_out)
            features.append(no_pad_features)
            # risks.append(no_pad_risk)
            patient_ids.append([patient]*len(no_pad_features))
            iterations.append([iteration]*len(no_pad_features))

    outputs = np.concatenate(outputs, axis=0)
    features = np.concatenate(features)
    patient_ids = np.concatenate(patient_ids)
    iterations = np.concatenate(iterations)
    # risks = np.concatenate(risks)
    
    return outputs, features, patient_ids, iterations

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
    emx = emx.where(np.triu(np.ones(emx.shape)).astype(np.bool))

    # Transform the matrix to a vector
    emx = emx.melt(id_vars=['id'])
    emx = emx.dropna()

    # Group feature interactions by the median
    emx = emx.groupby(['id', 'variable']).mean().reset_index()
    emx.columns = ['source', 'target', 'score']

    # Normalize feature interactions (cosine distance)
    emx['score'] = (emx.score - emx.score.min()) / (emx.score.max() - emx.score.min())
    emx['patient_id'] = patient

    return emx

def get_edges_importances(edges, dataset):
    edges_importance = []
    source_target = {}
    for ix, item in tqdm(edges.iterrows()):
        if item.source == item.target: continue

        try:
            assert(source_target[(item.source, item.target)])
            continue
        except:
            pass

        means = pd.concat([
            dataset[(dataset.source == item.source) & (dataset.target == item.target)],
            dataset[(dataset.source == item.target) & (dataset.target == item.source)]
        ])

        edges_importance.append(dict(
            source=item.source,
            target=item.target,
            score=means.score.mean(),
            score_std=means.score.std(),
            counts=means.shape[0],
            values=means.score.to_numpy()
        ))

        source_target[(item.source, item.target)] = True
        source_target[(item.target, item.source)] = True
    
    return pd.DataFrame(edges_importance)    

def feature_report(data, populations, feature, data_label='label', population_labels=['PR', 'UR'], axs=[]):
    
    Pop1, Pop2 = populations
    
    sns.kdeplot(
        data[(data.label == population_labels[0])][feature],
        color='black',
        fill=True,
        ax=axs[1],
        label=population_labels[0], 
        linewidth=2,
        hatch='///',
        alpha=0
    )
    
    sns.kdeplot(
        data[(data.label == population_labels[1])][feature],
        color='black',
        fill=True,
        ax=axs[1],
        label=population_labels[1], 
        linewidth=2,
        hatch='--',
        alpha=0
    )
    
    
    axs[1].set_title('Raw Data Distribution')
    
    sns.kdeplot(
        Pop1[Pop1.source == feature].score,
        color='black',
        fill=True,
        ax=axs[0],
        label=population_labels[0], 
        linewidth=2,
        hatch='///',
        alpha=0
    )
    
    sns.kdeplot(
        Pop2[Pop2.source == feature].score,
        color='black',
        fill=True,
        ax=axs[0],
        label=population_labels[1],
        linewidth=2,
        hatch='--',
        alpha=0
    )
    axs[0].set_title('Transformer Importance Distribution')
    axs[0].legend(loc='upper left')
    
    svr = PlotSurvival(data, time='OS_Months', censor='OS_Event')
    is_binary = True if len(list(set(data[feature]))) == 2 else False
    
    if not is_binary:
        q25 = np.quantile(data[feature], 0.25)
        q75 = np.quantile(data[feature], 0.75)

        svr.add(data[feature] <= q25, '{}'.format('Low Score'))
        svr.add((data[feature] > q25) & (data.β < q75), '{}'.format('Mid Score'))
        svr.add(data[feature] >= q75, '{}'.format('High Score'))

        svr.plot(
            axs[2], 
            ref = 'High Score', 
            targets=['Low Score', 'Mid Score', 'High Score'], 
            colors=['#5151bd', 'red', 'black'], 
            line_styles=['-', '-', '-'], 
            table=False, 
            plot=True, 
            legend=True,
            legend_font_size=8,
            label_font_size=8,
            lbox=[0.05, 0.01],
            legend_weight='bold',
            linewidth=2
        )

        axs[2].set_title('KM-Plot {}'.format(feature))
    
    else:
        lab1 = '{}'.format('0'.format())
        lab2 = '{}'.format('1'.format())

        svr.add(data[feature] == 0, lab1)
        svr.add((data[feature] == 1) , lab2)

        svr.plot(
            axs[2], 
            ref = lab1, 
            targets=[lab1, lab2], 
            colors=['#5151bd', 'black', 'black'], 
            line_styles=['-', '-', '-'], 
            table=False, 
            plot=True, 
            legend=True,
            legend_font_size=8,
            label_font_size=8,
            lbox=[0.05, 0.01],
            legend_weight='bold',
            linewidth=2
        )
        
        axs[2].set_title('KM-Plot {}'.format(feature))

def get_node_size(node_sizes, k):
    try:
        return node_sizes[k]
    except:
        return 0

def replace_names(db, key):
    try:
        return db[key]
    except:
        return key
    
def plot_feature_network(cssx, ax=None, edge_alpha=0.001, q=0.75, nodePos=False, draw=True, nodeFactor=100, less_than=False, pos=True, normalize=False, node_score='score_x', replace_db={}):
    
    cssx['source'] = [replace_names(replace_db, i) for i in cssx.source]
    cssx['target'] = [replace_names(replace_db, i) for i in cssx.target]
    
    node_sizes_df1 = cssx[cssx.source != cssx.target].groupby('source').mean().reset_index()
    node_sizes_df2 = cssx[cssx.source != cssx.target].groupby('target').mean().reset_index()
    node_sizes_df2['source'] = node_sizes_df2['target']
    
    node_sizes_df = pd.concat([node_sizes_df1, node_sizes_df2]).groupby('source').mean().reset_index()
    #display(node_sizes_df)
    
    if normalize:
        thr = np.quantile(cssx[cssx.source != cssx.target].score, q)
        cssx['score'] = (cssx['score'] - np.min(cssx['score'])) / (np.max(cssx['score']) - np.min(cssx['score']))
        cssx = cssx[(cssx.score >= thr)].reset_index(drop=True)
    
    if not ax:
        f=plt.figure(constrained_layout = True, figsize=(15, 5) )
        gs = f.add_gridspec(1, 3)
        ax = f.add_subplot(gs[0, 0])

    H = nx.from_pandas_edgelist(cssx, edge_attr='score')
    if not nodePos:
        nodePos = nx.circular_layout(H)

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

    node_size = {i.source: i[node_score] for ix,i in node_sizes_df.iterrows()}
    node_size_map = [nodeFactor*(get_node_size(node_size, i) + 0) for i in H.nodes()]
    
    edges = nx.get_edge_attributes(H,'score')
    edge_color = ['#abbab8' if H[i][j]['score'] > 0 else '#abbab8' for i,j in H.edges]
    
    node_color_map = [ 'lightgreen' for i in node_color_map_values]

    label_dict = {i:i.replace('molecular_', '') for i in H.nodes()}

    if draw:
        nx.draw(H, nodePos, node_color=node_color_map, node_size=node_size_map, alpha=0.91, width=0.0, ax=ax, with_labels=True, font_size=12, labels=label_dict, font_weight="bold")
        nx.draw_networkx_edges(H, nodePos, edgelist=list(edges.keys()), alpha=0.8, width=(edge_alpha*np.array(list(edges.values()))), ax=ax, edge_color=edge_color); 
    
    return nodePos, node_size, H

def pairwise_feature_report(data, populations, feature1, feature2, data_label='label', population_labels=['PR', 'UR'], axs=[]):

    Pop1, Pop2 = populations
    
    sns.kdeplot(
        Pop1[(Pop1.source == feature1) & (Pop1.target == feature2)].score,
        color='black',
        fill=True,
        ax=axs[0],
        label=population_labels[0], 
        linewidth=2,
        hatch='///',
        alpha=0
    )
    
    sns.kdeplot(
        Pop2[(Pop2.source == feature1) & (Pop2.target == feature2)].score,
        color='black',
        fill=True,
        ax=axs[0],
        label=population_labels[1],
        linewidth=2,
        hatch='--',
        alpha=0
    )
    
    stat, p = stats.mannwhitneyu(
        Pop1[(Pop1.source == feature1) & (Pop1.target == feature2)].score, 
        Pop2[(Pop2.source == feature1) & (Pop2.target == feature2)].score
    )
    
    axs[0].set_title('Importance Distribution\n{}, {} ({:.2e})'.format(feature1, feature2, p))
    axs[0].legend(loc='upper left')
    axs[0].set_xlim([0, 1])

def load_data(fi, path, run, epoch, sample_id='SAMPLE_ID', time='OS_MONTHS', event='OS_STATUS', iterations=50):
    
    # load data
    data = pd.read_csv(fi, sep=',')

    # load model
    trainer = load_model(path, run, epoch=epoch)
    transformed_data = trainer.data_converter.transform(data)
    evaluator = SurvivalEstimator(model=trainer, path=path, run=run, epoch=epoch)

    # Predicting risk scores
    β, w, o, t, r = evaluator.predict(
            transformed_data, iterations=iterations, normalize=True, batch_size=10000
        )
    
    β = β.mean(axis=0)
    data['β'] = β[:, 0]

    y=np.array(data[[time, event]])
    cindex = sigmoid_concordance(y, β)

    # Adding cutoffs for populations
    q25 = np.quantile(data.β, 0.25)
    q75 = np.quantile(data.β, 0.75)

    data['label'] = '-'
    data.loc[data.β <= q25, 'label'] = '<25th'
    data.loc[data.β >= q75, 'label'] = '>75th'

    # Obtaining attention scores per patient
    outputs, features, patient_ids, iters = extract_patient_embeddings(data, sample_id, o, t, r, iterations)

    return data, cindex, outputs, features, patient_ids, iters, q25, q75

def compute_attention_scores_app(data, features, outputs, iterations, patient_ids, sample_id='SAMPLE_ID'):
    emxso = []
    for pix, patient in tqdm(data.iterrows(), total=data.shape[0]):
        emx = inner_cosine_scores(patient[sample_id], features, outputs, iterations, patient_ids)
        emx['β'] = patient.β
        
        emxso.append(emx)
    
    emxs = pd.concat(emxso)
    return emxs

def extract_edges_from_attention_scores(emxsp, q25, q75):

    PR_pop = emxsp[emxsp.β <= q25].reset_index(drop=True)
    UR_pop = emxsp[emxsp.β >= q75].reset_index(drop=True)

    edges_UR = UR_pop.groupby(['source', 'target']).count().reset_index()[['source', 'target', 'score']]
    edges_PR = PR_pop.groupby(['source', 'target']).count().reset_index()[['source', 'target', 'score']]

    UR_edges = get_edges_importances(edges_UR[edges_UR.score > 10], UR_pop)
    PR_edges = get_edges_importances(edges_PR[edges_UR.score > 10], PR_pop)  

    URPR = pd.merge(PR_edges[PR_edges.counts > 1], UR_edges[UR_edges.counts > 1], on=['source', 'target'], how='outer').fillna(1)
    URPR['UR-PR'] = URPR.score_y - URPR.score_x
    URPR['UR-PR'] = stats.zscore( URPR['UR-PR'] )

    pvals = []
    for ix, i in URPR.iterrows():
        try:
            t = stats.mannwhitneyu(np.array(i.values_x), np.array(i.values_y) )
            pvals.append(-np.log10(t[1]) )
        except:
            pvals.append(0)

    URPR['pval'] = pvals

    return UR_edges, PR_edges, UR_pop, PR_pop, URPR

def filter_edges(URPR, pval, lqt=0.25, uqt=0.75):
    URPR = URPR.copy()[URPR.pval > pval].reset_index(drop=True)

    URPR_uqt = np.quantile(URPR['UR-PR'], uqt)
    URPR_lqt = np.quantile(URPR['UR-PR'], lqt)

    edges_UR = URPR[(URPR['UR-PR'] >= URPR_uqt)].reset_index(drop=True)
    edges_PR = URPR[(URPR['UR-PR'] < URPR_lqt)].reset_index(drop=True)

    edges_UR['score'] = np.abs(edges_UR['UR-PR'])
    edges_PR['score'] = np.abs(edges_PR['UR-PR'])

    f, axs = subplots(w=10, h=5, cols=2, rows=1, return_f=True)

    percentile = 0.0
    replace_nodes = {}

    URPR['score'] = 1
    nodePos,size, H = plot_feature_network(
        URPR.copy(), 
        q=percentile, ax=axs[0], draw=False,
        replace_db=replace_nodes
    )

    _,size, H = plot_feature_network(
        edges_PR.copy(), 
        q=percentile, ax=axs[0], nodePos=nodePos, edge_alpha=2, draw=True, nodeFactor=10, pos=False,
        normalize=True,
        node_score='pval',
        replace_db=replace_nodes
    )
    axs[0].set_title('β<25th')

    _,size, H = plot_feature_network(
        edges_UR.copy(), 
        q=percentile, ax=axs[1], nodePos=nodePos, edge_alpha=2, draw=True, nodeFactor=10, pos=False,
        normalize=False,
        node_score='pval',
        replace_db=replace_nodes
    )
    axs[1].set_title('β>75th')

    st.pyplot(f)

def interaction_distributions(data, PR_pop, UR_pop, emxs, feature1, feature2, cutoff=0.5, sample_id='', time='', event='', ylim=[]):
    f, axs = subplots(cols=2, rows=1, h=4, w=10, return_f=True)
    pairwise_feature_report(
        data, 
        [PR_pop, UR_pop],
        feature1=feature1,
        feature2=feature2,
        axs=axs,
        population_labels=['β<25th', 'β>75th'],
    )

    axs[0].axvline(x=cutoff, color='black')
    axs[0].axvspan(cutoff, 1, alpha=0.2, color='red')
    axs[0].axvspan(0, cutoff, alpha=0.1, color='#5151bd')
    axs[0].set_xlabel('Feature Interaction Score')

    # emxsp = emxs[ ~(emxs.source.str.match('molecular_') | emxs.target.str.match('molecular_')) ].reset_index(drop=True).copy()
    emxsp = emxs.reset_index(drop=True).copy()
    pt = emxsp[(emxsp.source == feature1) & (emxsp.target == feature2)].groupby('patient_id').mean().reset_index()[['patient_id', 'score']]
    positive = list(set(pt[pt.score > cutoff].patient_id))

    svr = PlotSurvival(data, time=time, censor=event)

    svr.add(data[sample_id].isin(positive), '{}'.format('Population'))
    svr.add(~data[sample_id].isin(positive), '{}'.format('Background'))

    svr.plot(
        axs[1], 
        ref = 'Background', 
        targets=['Background', 'Population'], 
        colors=['#5151bd', 'red', 'black'], 
        line_styles=['-', '-', '-'], 
        table=True, 
        plot=True, 
        legend=True,
        legend_font_size=10,
        label_font_size=14,
        lbox=[0.01, 0.01],
        legend_weight='normal',
        linewidth=1.5
    )

    st.pyplot(f)

    f, axs = subplots(w=15, h=5, cols=1, rows=1, return_f=True)

    rlab = np.array(['Background']*data.shape[0])
    rlab[data[sample_id].isin(positive)] = 'Population'

    palette = {'Background': '#5151bd', 'Population': 'red'}
    
    try:
        x = np.array(data[feature2])
        xi = np.array([None]*len(x))
        pqsi = 0
        xorder = []
        for qx, q in enumerate([0.25, 0.5, 0.75]):
            qsi = np.quantile(data[feature2], q)
            label = "[{:.1f}-{:.1f}]".format(pqsi, qsi)
            xorder.append(label)
            
            xi[(x <= qsi) & (x!=None)] = label
            x[(x <= qsi)] = None
            pqsi = qsi

        xi[xi == None] = ">{:.1f}".format(qsi)
        xorder.append(">{:.1f}".format(qsi))

        x = xi

    except:
        xorder = []
        x = data[feature2]

    y = data[feature1]

    if len(xorder) > 0:
        sns.stripplot(x=x, y=y, hue=rlab, ax=axs[0], palette=palette, size=10, dodge=True, edgecolor='white', order=xorder, linewidth=1, jitter=True)
        sns.boxplot(x=x, y=y, hue=rlab, ax=axs[0], palette=palette, order=xorder)
    else: 
        sns.boxplot(x=x, y=y, hue=rlab, ax=axs[0], palette=palette)
        sns.stripplot(x=x, y=y, hue=rlab, ax=axs[0], palette=palette, size=10, alpha=0.85, dodge=True, edgecolor='black', linewidth=1)

    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha='right', )
    if len(ylim)>0: axs[0].set_ylim(ylim)
    st.pyplot(f)

def survival_plot(data, q25, q75, time, event): 
    svr = PlotSurvival(data, time=time, censor=event)

    svr.add(data.label == '<25th', '{}'.format('<25th'))
    svr.add(data.label == '-', '{}'.format('-'))
    svr.add(data.label == '>75th', '{}'.format('>75th'))

    f, axs = subplots(cols=2, rows=1, w=10, h=5, return_f=True)

    svr.plot(
        axs[1], 
        ref = '<25th', 
        targets=['>75th', '-', '<25th'], 
        colors=['#5151bd', 'red', 'black'], 
        line_styles=['-', '-', '-'], 
        table=True, 
        plot=True, 
        legend=True,
        legend_font_size=10,
        label_font_size=10,
        lbox=[0.04, 0.01],
        legend_weight='bold',
        linewidth=2
    )
    
    sns.kdeplot(data.β, ax=axs[0], fill=True)
    axs[0].axvline(x=q25, color='black')
    axs[0].axvline(x=q75, color='#5151bd');
    
    st.pyplot(f)

def compute_interaction_scores(URPR):
    interaction_score = []
    for variable in set(list(set(URPR.source)) + list(set(URPR.target))):
        z = URPR[(URPR.source == variable) | (URPR.target == variable)].reset_index(drop=True)
        
        pval = ttest_ind(z.UR_score_mean, z.PR_score_mean)[1]
        
        x1 = np.mean(z.UR_score_mean)
        x2 = np.mean(z.PR_score_mean)
        
        interaction_score.append({
            'source': variable,
            'pval': pval,
            'UR-PR': x1 - x2,
            'degree': z.shape[0]
        })
    
    return pd.DataFrame(interaction_score)

def observations(x):
    return len(x)

def ttest(i):
    try:
        return ttest_ind_from_stats(i.UR_score_mean, i.UR_score_std, i.UR_score_observations, i.PR_score_mean, i.PR_score_std, i.PR_score_observations, )[1]
    except:
        return np.nan

def fix_edge_names(emxs):
    urd = {}
    sources = []
    targets = []
    for ix, i in tqdm(emxs.iterrows(), total=emxs.shape[0]):
        try:
            if urd[(i.source, i.target)] == 1:
                sources.append(i.target)
                targets.append(i.source)
            else:
                sources.append(i.source)
                targets.append(i.target)
        except:
            urd[(i.source, i.target)] = 0
            urd[(i.target, i.source)] = 1

            sources.append(i.source)
            targets.append(i.target)

    emxs['source'] = sources
    emxs['target'] = targets

def extract_populations(attention_scores, q25, q75):
    PR_pop = attention_scores[attention_scores.β <= q25].reset_index(drop=True)
    UR_pop = attention_scores[attention_scores.β >= q75].reset_index(drop=True)
    
    UR = UR_pop.groupby(['source', 'target']).agg({'score': [np.median, np.mean, np.std, observations]}).reset_index().fillna(0)
    UR.columns = [ "UR_{}".format("_".join(i)) if i[1] != '' else i[0] for i in UR.columns.to_flat_index()]

    PR = PR_pop.groupby(['source', 'target']).agg({'score': [np.median, np.mean, np.std, observations]}).reset_index().fillna(0)
    PR.columns = [ "PR_{}".format("_".join(i)) if i[1] != '' else i[0] for i in PR.columns.to_flat_index()]
    
    URPR = pd.merge(UR, PR, on=['source', 'target'])
    URPR['pval'] = [ ttest(i) for ix, i in URPR.iterrows()]
    URPR = URPR[URPR.source != URPR.target].reset_index(drop=True)
    
    URPR['UR-PR'] = URPR.UR_score_mean - URPR.PR_score_mean
    URPR.dropna(inplace=True)
    
    return URPR

def extract_populations_time(attention_scores, q25, q75, time):
    PR_pop = attention_scores[attention_scores[time] <= q25].reset_index(drop=True)
    UR_pop = attention_scores[attention_scores[time] >= q75].reset_index(drop=True)
    
    UR = UR_pop.groupby(['source', 'target']).agg({'score': [np.median, np.mean, np.std, observations]}).reset_index().fillna(0)
    UR.columns = [ "UR_{}".format("_".join(i)) if i[1] != '' else i[0] for i in UR.columns.to_flat_index()]

    PR = PR_pop.groupby(['source', 'target']).agg({'score': [np.median, np.mean, np.std, observations]}).reset_index().fillna(0)
    PR.columns = [ "PR_{}".format("_".join(i)) if i[1] != '' else i[0] for i in PR.columns.to_flat_index()]
    
    URPR = pd.merge(UR, PR, on=['source', 'target'])
    URPR['pval'] = [ ttest(i) for ix, i in URPR.iterrows()]
    URPR = URPR[URPR.source != URPR.target].reset_index(drop=True)
    
    URPR['UR-PR'] = URPR.UR_score_mean - URPR.PR_score_mean
    URPR.dropna(inplace=True)
    
    return URPR

import networkx as nx
def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % (rgb[0], 0, rgb[2])

def feature_interaction_graph(interaction_score, URPR, lower_thr=0.25, upper_thr=0.75, draw=False, axs=[]):
    
    GPR = nx.Graph()
    GUR = nx.Graph()
    Gp = nx.Graph()
    for ix,i in URPR.iterrows():
        if i.pval < 0.05:
            if i['UR-PR'] < np.quantile(URPR['UR-PR'], lower_thr):
                GPR.add_edge(i.source, i.target, weight=np.abs(i['UR-PR']) )
            elif i['UR-PR'] > np.quantile(URPR['UR-PR'], upper_thr):
                GUR.add_edge(i.source, i.target, weight=i['UR-PR'])

        Gp.add_edge(i.source, i.target, weight=i['UR-PR'])
    
    if draw:
        
        pos = nx.circular_layout(Gp)

        labels = nx.draw_networkx_labels(GPR, pos=pos, ax=axs[1])
        weights = nx.get_edge_attributes(GPR,'weight')
        edges = GPR.edges()
        colors = [rgb_to_hex(tuple(np.repeat(int(255 * (1- weights[edge])),3))) for edge in edges]
        nx.draw(
            GPR, 
            pos=pos, ax=axs[1], 
            width=10*np.array(list(weights.values())), 
            edge_color=colors,
            node_color='white',
            node_size=1
        )
        axs[1].set_title('PR')

        labels = nx.draw_networkx_labels(GUR, pos=pos, ax=axs[2])
        weights = nx.get_edge_attributes(GUR,'weight')
        edges = GUR.edges()
        colors = [rgb_to_hex(tuple(np.repeat(int(255 * (1- weights[edge])),3))) for edge in edges]
        nx.draw(
            GUR, 
            pos=pos, ax=axs[2], 
            width=10*np.array(list(weights.values())), 
            edge_color=colors,
            node_color='white',
            node_size = 1
        ) 
        axs[2].set_title('UR');


        sns.barplot(
            data=interaction_score.sort_values('UR-PR'),
            y='source',
            x='UR-PR',
            ax=axs[0]
        );

        axs[0].set_title('Feature Interaction Score');
    
    return GPR, GUR