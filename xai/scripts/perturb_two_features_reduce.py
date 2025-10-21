import sys
import yaml

config = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)

lib=config['lib']
run_id=config['run_id']
prem=config['prem']
epoch=config['epoch']
root_dir=config['root_dir']
time=config['time']
event=config['event']
sample_id=config['sample_id']
fold=config['fold']
fold_id=config['fold_id']
features=config['features']
study=config['study']
sample_size=config['sample_size']
items_per_partition=config['items_per_partition']
cores=config['cores']
data_size=config['data_size']
batch_size=config['batch_size']
var1=config['var1']
var2=config['var2']
sys.path = lib


from samecode.plot.pyplot import subplots
from samecode.survival.plot import PlotSurvival

from tqdm.auto import tqdm
from scipy import stats
from collections import Counter

from sklearn.model_selection import train_test_split
from xai.models.explainer import TransformerSurvivalEvaluator
from xai.models.explainer import survival_attention_scores
from xai.models.SimplifiedClinicalTransformer.utils import load_model

from sklearn.model_selection import train_test_split
from xai.models.explainer import SurvivalExtractor

import os
import pandas as pd
from samecode.random import set_seed
import numpy as np 
import pickle
import multiprocessing

from samecode.logger.logger import logger
from samecode.preprocess import split_vector


logger = logger()

path = '{}/{}/artifacts/models/{}/'.format(root_dir, run_id, prem)
runs = [i for i in os.listdir(path) if 'fold-' in i]

logger.info('Reduce')
splits = [[var1, var2, i[0], i[1], i[2]] for i in split_vector(data_size, items_per_partition)]

pts = []
for var1, var2, start, stop, index in splits:
	fo = '{}/{}/generative_model_E{}/Perturbed_Feature_{}_and_Feature_{}/split-{}.pk'.format(path, fold_id, epoch, var1, var2, index).replace(' ', '_')
	
	pts.append(
		pickle.load(open(fo, 'rb'))
	)
	
	os.system('rm {}'.format(fo))

logger.info('saving ...')
pts = pd.concat(pts)
fo = '{}/{}/generative_model_E{}/Perturbed_Feature_{}_and_Feature_{}/generated.pk'.format(path, fold_id, epoch, var1, var2).replace(' ', '_')
pickle.dump(pts, open(fo, 'wb'))