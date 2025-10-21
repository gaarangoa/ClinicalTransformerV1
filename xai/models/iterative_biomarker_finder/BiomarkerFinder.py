import nevergrad as ng
from concurrent import futures
import pandas as pd
import numpy as np 
from judge.experimental.survival import PlotSurvival
import sys
import os 
from collections import Counter
import logging
from sklearn.utils import resample
import pickle 

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

class PredictiveBiomarkerFinder():
    def __init__(self, data, arm, treatments=['Durva', 'D+T', 'SOC'], time='OS', censor='event',):
        self.data = data
        
        self.arm = arm
        self.treatments = treatments
        self.time = time
        self.censor = censor
        self.history = []
    
    def objective_function(self, *args, **kwargs):
        
        self.iter += 1
        sys.stdout.write("\riter: {}".format(self.iter))
        sys.stdout.flush()
        
        qvars = [kwargs['{}'.format(i)] for i in self.variables]
        
        if not self.conditions:
            qconds = [kwargs['{}_condition'.format(i)] for i in self.variables]
        else:
            qconds = {i: j for i,j in zip(self.variables, self.conditions)}
            
        svr = PlotSurvival(self.data, time=self.time, censor=self.censor)
        for treatment in self.treatments:
            
            _expression = []
            for vx, var in enumerate(self.variables):
                conditional_value = np.quantile(np.array(self.data[self.data[self.arm] == treatment][self.variables[vx]]), qvars[vx]/100)
                
                if not self.conditions:
                    val_condition = "(self.data['{}'] {} {}) ".format(self.variables[vx], qconds[vx], conditional_value)
                else: 
                    val_condition = "(self.data['{}'] {} {}) ".format(self.variables[vx], qconds[var], conditional_value)
                
                _expression.append(val_condition)
            
            ix = eval( '&'.join(_expression) )
            
            cx = np.array(list(Counter(ix).values()))
        
            if (treatment == self.target_treatment) & (np.min(cx) / np.sum(cx) <= self.min_population):
                return 1
            
            try:
                svr.add(ix & (self.data[self.arm] == treatment), '{} (H)'.format(treatment))
                svr.add(~ix & (self.data[self.arm] == treatment), '{} (L)'.format(treatment))
            except:
                return 1

        ref_ = '{} (H)'.format(self.baseline_treatment)
        tar  = '{} (H)'.format(self.target_treatment)
        svr.plot([], ref = ref_, targets=[ref_, tar], colors=['black', '#5151bd'], line_styles=['-', '-'], table=False, plot=False, legend=False)

        perf = {
                "Reference": ref_, 
                'Target': tar, 
                'HR': svr.kmfs[tar][3][0], 
                'Pvalue': svr.kmfs[tar][3][1], 
                'CI_Low': svr.kmfs[tar][3][2], 
                'CI_High': svr.kmfs[tar][3][3], 
                'Reference_population': int(svr.counts[0.0][ref_]),
                'Target_population': int(svr.counts[0.0][tar]),
            }
        
        perf.update(kwargs)
        
        self.history.append(perf)
        
        population = self.data[self.data[self.arm] == self.target_treatment].shape[0]
        population_score = (perf['Target_population'] / population)

        if population_score <= self.min_population:
            return 1
    
        if (perf['HR'] > 1):
            return 1
        else:
            return perf['HR']
                
    def optimize(self, baseline_treatment='SOC', target_treatment='D+T', conditions=[], budget=500, num_workers=10, min_population=0.3, variables=['counts_somatic', 'ratio', 'somatic_fraction'], initial_values=[]):
        
        self.variables=variables
        self.iter = 0
        
        if not conditions:
            self.parametrization = eval('ng.p.Instrumentation({})'.format(','.join( ['{parameter}=ng.p.Scalar(lower=5, upper=95), {parameter}_condition=ng.p.Choice([">=", "<="])'.format(parameter=i) for i in self.variables ])))
            self.conditions = []
        else: 
            self.parametrization = eval('ng.p.Instrumentation({})'.format(','.join( ['{}=ng.p.Scalar(lower=5, upper=95, )'.format(i) for i in self.variables ])))
            self.conditions = conditions
            
        
        self.min_population = min_population
        self.baseline_treatment=baseline_treatment
        self.target_treatment=target_treatment
        
        if initial_values: 
            self.parametrization.values = {i:j for i, j in zip(self.variables, initial_values)}
        
        self.optimizer = ng.optimizers.PSO(parametrization=self.parametrization, budget=budget, num_workers=num_workers)
        
        with futures.ThreadPoolExecutor(max_workers=self.optimizer.num_workers) as executor:  # the executor will evaluate the function in multiple threads
            self.recommendation = self.optimizer.minimize(self.objective_function, executor=executor)
   
    def plot(self, data, percentiles, axs, reference, target): 
        
        percentiles = [percentiles[i] for i in self.variables]
        
        svr = PlotSurvival(data, time=self.time, censor=self.censor)
        for treatment in self.treatments:

            _expression = []
            for vx, [[var, vqt], [var_condition, vcd]] in enumerate( zip( [(i, j) for i,j in zip(self.variables, percentiles)],  [(i, j) for i,j in zip(self.variables, self.conditions)]) ):

                conditional_value = np.quantile(np.array(data[data[self.arm] == treatment][var]), vqt/100)
                val_condition = "(data['{}'] {} {}) ".format(var, vcd, conditional_value)
                _expression.append(val_condition)

            ix = eval( '&'.join(_expression) )

            svr.add(ix & (data[self.arm] == treatment), '{} (H)'.format(treatment))
            svr.add(~ix & (data[self.arm] == treatment), '{} (L)'.format(treatment))

        
        ref_ = '{} (H)'.format(reference)
        tar2 = '{} (H)'.format(target)
        svr.plot(axs, ref = ref_, targets=[ref_, tar2], colors=['black', '#5151bd', 'red'], line_styles=['-', '-', '-'], table=True, plot=True, legend=True)


        for vx, [[var, cut], [var_condition, vcd]] in enumerate( zip( [(i, j) for i,j in zip(self.variables, percentiles)],  [(i, j) for i,j in zip(self.variables, self.conditions)]) ):
            ax = f.add_subplot(gs[vx, 3:4])
            sns.kdeplot(data[data[self.arm] == target][var], ax=ax, fill=True);

            for axis in ['top','right', 'left', 'bottom']:
                ax.spines[axis].set_linewidth(0.0)
                th = np.quantile(np.array(data[data[self.arm] == target][var]), cut/100)
                ax.axvline(th, c='black', linestyle='--')
                ax.text(1.2*th, 0.9*ax.get_ylim()[1], s="{}{:.2f} ({:.0f}th)".format(vcd, th, cut), weight='normal', color='#5151bd', ha='left')
                ax.get_yaxis().set_visible(False)
        
def PredictiveOptimizerBootstrap(dataset, random_seed, train_fraction_population, variables, arm, treatments, time, censor, baseline_treatment, target_treatment, budget, num_workers, min_population, initial_values=[], conditions=[], iterations=10, outfile=''):
    ''' 
    dataset=dataset, 
    random_seed=0,
    train_fraction_population=0.8,
    variables=['bTMB', 'somatic_fraction'], 
    arm='ACTARM_trunc', 
    treatments=['SOC', 'IO'], 
    time='OS', 
    censor='event', 
    baseline_treatment='SOC', 
    target_treatment='IO', 
    budget=512, 
    num_workers=200, 
    min_population=0.3, 
    initial_values=[], 
    conditions=['>=', '>='], 
    iterations=100, 
    outfile='./optim/iter-0'
    
    '''
    dataset['_cid'] = range(dataset.shape[0])
    feat = variables
    
    perf = []
    H = []
    for _sedx in range(iterations):
        logger.info('Iteration: {} of {}'.format(_sedx, iterations))
        train = resample(
            dataset, 
            n_samples= int(train_fraction_population*dataset.shape[0]) , 
            replace=False, 
            stratify=dataset.fillna(0)[arm],
            random_state = 10000000*(random_seed + 1) + _sedx
        )

        test = dataset[~dataset._cid.isin(train._cid)]

        optimizer = PredictiveBiomarkerFinder(train.fillna(0), arm=arm, time=time, censor=censor, treatments=treatments)

        optimizer.optimize(
            baseline_treatment=baseline_treatment, 
            target_treatment=target_treatment, 
            variables=feat,
            conditions=conditions,
            budget=budget, 
            min_population=min_population,
            num_workers=num_workers,
        )

        h = pd.DataFrame(optimizer.history).sort_values(['HR', 'Pvalue'], ascending=[True, True])
        H.append(h)
        
        data = test.fillna(0)
        exp = h[:1].index[0]

        svr = PlotSurvival(data, time='OS', censor='event')
        for treatment in treatments:

            _expression = []
            for vx, [[var, vqt], [var_condition, vcd]] in enumerate( zip( h[['{}'.format(i) for i in feat]].loc[exp].to_dict().items(),  [(i, j) for i,j in zip(feat, conditions)]) ):

                conditional_value = np.quantile(np.array(data[data[arm] == treatment][var]), vqt/100)
                val_condition = "(data['{}'] {} {}) ".format(var, vcd, conditional_value)
                _expression.append(val_condition)

            ix = eval( '&'.join(_expression) )

            svr.add(ix & (data[arm] == treatment), '{} (H)'.format(treatment))
            svr.add(~ix & (data[arm] == treatment), '{} (L)'.format(treatment))

        ref_ = '{} (H)'.format(baseline_treatment)
        tar = '{} (H)'.format(target_treatment)
        svr.plot([], ref = ref_, targets=[ref_, tar], colors=['black', '#5151bd', 'red'], line_styles=['-', '-', '-'], table=False, plot=False, legend=False)

        perf_ = {
                    "Reference": ref_, 
                    'Target': tar, 
                    'HR': svr.kmfs[tar][3][0], 
                    'Pvalue': svr.kmfs[tar][3][1], 
                    'CI_Low': svr.kmfs[tar][3][2], 
                    'CI_High': svr.kmfs[tar][3][3], 
                    'Reference_population': int(svr.counts[0.0][ref_]),
                    'Target_population': int(svr.counts[0.0][tar]),
                }

        perf_.update({k: h[:1][k] for k in feat})
        perf.append(perf_)
    
    pickle.dump(
        {"Performance": perf, "Training": H},
        open('{}.pk'.format(outfile), 'wb')
    )


    return {"Performance": perf, "Training": H}

class PrognosticBiomarkerFinder():
    def __init__(self, data, arm, treatments=['SOC'], time='OS', censor='event',):
        self.data = data
        
        self.arm = arm
        self.treatments = treatments
        self.time = time
        self.censor = censor
        self.history = []
        self.counter = 0
    
    def objective_function(self, *args, **kwargs):
        
        self.counter += 1 
        qvars = [kwargs['{}'.format(i)] for i in self.variables]
        
        if not self.conditions:
            qconds = [kwargs['{}_condition'.format(i)] for i in self.variables]
        else:
            qconds = {i: j for i,j in zip(self.variables, self.conditions)}
        
        svr = PlotSurvival(self.data, time=self.time, censor=self.censor)
        for treatment in self.treatments:
            
            _expression = []
            for vx, var in enumerate(self.variables):
                conditional_value = np.quantile(np.array(self.data[self.data[self.arm] == treatment][self.variables[vx]]), qvars[vx]/100 )
                
                if not self.conditions:
                    val_condition = "(self.data['{}'] {} {}) ".format(self.variables[vx], qconds[vx], conditional_value)
                else: 
                    val_condition = "(self.data['{}'] {} {}) ".format(self.variables[vx], qconds[var], conditional_value)
                    
                _expression.append(val_condition)
            
            ix = eval( '&'.join(_expression) )
            
            cx = np.array(list(Counter(ix).values()))
        
            if (treatment == self.baseline_treatment) & (np.min(cx) / np.sum(cx) <= self.min_population):
                return 1
            
            try:
                svr.add(ix & (self.data[self.arm] == treatment), '{} (H)'.format(treatment))
                svr.add(~ix & (self.data[self.arm] == treatment), '{} (L)'.format(treatment))
            except Exception as inst:
                return 1

        ref_ = '{} (H)'.format(self.baseline_treatment)
        tar  = '{} (L)'.format(self.baseline_treatment)
        
        try:
            svr.plot([], ref = ref_, targets=[ref_, tar], colors=['black', '#5151bd'], line_styles=['-', '-'], table=False, plot=False, legend=False)
        except:
            return 1
        
        perf_1 = {
                "Reference": ref_, 
                'Target': tar, 
                'HR': svr.kmfs[tar][3][0], 
                'Pvalue': svr.kmfs[tar][3][1], 
                'CI_Low': svr.kmfs[tar][3][2], 
                'CI_High': svr.kmfs[tar][3][3], 
                'Reference_population': cx[0],
                'Target_population': cx[1],
            }
        
        
        ref_ = '{} (L)'.format(self.baseline_treatment)
        tar  = '{} (H)'.format(self.baseline_treatment)
        
        try:
            svr.plot([], ref = ref_, targets=[ref_, tar], colors=['black', '#5151bd'], line_styles=['-', '-'], table=False, plot=False, legend=False)
        except:
            return 1
        
        perf_2 = {
                "Reference": ref_, 
                'Target': tar, 
                'HR': svr.kmfs[tar][3][0], 
                'Pvalue': svr.kmfs[tar][3][1], 
                'CI_Low': svr.kmfs[tar][3][2], 
                'CI_High': svr.kmfs[tar][3][3], 
                'Reference_population': cx[0],
                'Target_population': cx[1],
            }
        
        if perf_1['HR'] > perf_2['HR']:
            perf = perf_2
        else:
            perf = perf_1
        
        perf.update(kwargs)

        self.history.append(perf)
        
        sys.stdout.write("\riter: {} Loss: {:.2f} Ref: {} Tar: {}".format(self.counter, perf['HR'] + np.abs(np.log(perf['Reference_population'] / perf['Target_population'])), perf['Reference_population'], perf['Target_population']))
        sys.stdout.flush()
        
        
        return perf['HR'] # + np.abs(np.log(perf['Reference_population'] / perf['Target_population']))
    
    def optimize(self, baseline_treatment='SOC', budget=500, num_workers=10, min_population=0.3, variables=['counts_somatic', 'ratio', 'somatic_fraction'], conditions=[]):
        
        self.variables=variables
        self.counter = 0
        
        
        if not conditions:
            self.parametrization = eval('ng.p.Instrumentation({})'.format(','.join( ['{parameter}=ng.p.Scalar(lower=5, upper=98), {parameter}_condition=ng.p.Choice([">=", "<="])'.format(parameter=i) for i in self.variables ])))
            self.conditions = []
        else: 
            self.parametrization = eval('ng.p.Instrumentation({})'.format(','.join( ['{}=ng.p.Scalar(lower=5, upper=98, )'.format(i) for i in self.variables ])))
            self.conditions = conditions
        
        
        self.min_population = min_population
        self.baseline_treatment=baseline_treatment
        
        self.optimizer = ng.optimizers.PSO(parametrization=self.parametrization, budget=budget, num_workers=num_workers)
        
        with futures.ThreadPoolExecutor(max_workers=self.optimizer.num_workers) as executor:  # the executor will evaluate the function in multiple threads
            self.recommendation = self.optimizer.minimize(self.objective_function, executor=executor)

    def plot(self, data, percentiles, axs, reference, target):
        
        percentiles = [percentiles[i] for i in self.variables]
        
        svr = PlotSurvival(data, time=self.time, censor=self.censor)
        for treatment in self.treatments:

            _expression = []
            for vx, [[var, vqt], [var_condition, vcd]] in enumerate( zip( [(i, j) for i,j in zip(self.variables, percentiles)],  [(i, j) for i,j in zip(self.variables, self.conditions)]) ):

                conditional_value = np.quantile(np.array(data[data[self.arm] == treatment][var]), vqt/100)
                val_condition = "(data['{}'] {} {}) ".format(var, vcd, conditional_value)
                _expression.append(val_condition)

            ix = eval( '&'.join(_expression) )

            svr.add(ix & (data[self.arm] == treatment), '{} (H)'.format(treatment))
            svr.add(~ix & (data[self.arm] == treatment), '{} (L)'.format(treatment))

        
        ref_ = '{} (H)'.format(reference)
        tar2 = '{} (L)'.format(target)
        svr.plot(axs, ref = ref_, targets=[ref_, tar2], colors=['black', '#5151bd', 'red'], line_styles=['-', '-', '-'], table=True, plot=True, legend=True)


        for vx, [[var, cut], [var_condition, vcd]] in enumerate( zip( [(i, j) for i,j in zip(self.variables, percentiles)],  [(i, j) for i,j in zip(self.variables, self.conditions)]) ):
            ax = f.add_subplot(gs[vx, 3:4])
            sns.kdeplot(data[data[self.arm] == target][var], ax=ax, fill=True);

            for axis in ['top','right', 'left', 'bottom']:
                ax.spines[axis].set_linewidth(0.0)
                th = np.quantile(np.array(data[data[self.arm] == target][var]), cut/100)
                ax.axvline(th, c='black', linestyle='--')
                ax.text(1.2*th, 0.9*ax.get_ylim()[1], s="{}{:.2f} ({:.0f}th)".format(vcd, th, cut), weight='normal', color='#5151bd', ha='left')
                ax.get_yaxis().set_visible(False)

def PrognosticOptimizerBootstrap(dataset, random_seed, train_fraction_population, variables, arm, time, censor, treatment, budget, num_workers, min_population, initial_values=[], conditions=[], iterations=10, outfile=''):
    ''' 
    perf, track = PrognosticOptimizerBootstrap(
        dataset=data, 
        random_seed=0,
        train_fraction_population=0.5,
        variables=['x_1', 'x_2', 'x_3'],
        arm='ARM', 
        time='time', 
        censor='event', 
        treatment='SOC', 
        budget=1024, 
        num_workers=10, 
        min_population=0.3, 
        initial_values=[], 
        iterations=1, 
        outfile='./optim/iter-0'
    )
    
    '''
    dataset['_cid'] = range(dataset.shape[0])
    feat = variables
    
    try:
        os.system('mkdir -p {}'.format(outfile))
    except:
        pass

    perf = []
    H = []
    for _sedx in range(iterations):
        logger.info('Iteration: {} of {}'.format(_sedx, iterations))
        train = resample(
            dataset, 
            n_samples= int(train_fraction_population*dataset.shape[0]) , 
            replace=False, 
            stratify=dataset.fillna(0)[arm],
            random_state = 10000000*(random_seed + 1) + _sedx
        )

        test = dataset[~dataset._cid.isin(train._cid)]
        
        optimizer = PrognosticBiomarkerFinder(train.fillna(0).reset_index(drop=True), arm=arm, time=time, censor=censor, treatments=[treatment])

        optimizer.optimize(
            baseline_treatment=treatment, 
            variables=feat,
            conditions=conditions,
            budget=budget, 
            min_population=min_population,
            num_workers=num_workers,
        )

        h = pd.DataFrame(optimizer.history).sort_values(['HR', 'Pvalue'], ascending=[True, True]).reset_index(drop=True)
        H.append(h)

        data = test.fillna(0)
        exp = h[:1].index[0]
        
        if conditions:
            best_model_ = list(zip( h[['{}'.format(i) for i in feat]].loc[exp].to_dict().items(),  [(i, j) for i,j in zip(feat, conditions)]))
        else:
            best_model_ = h.loc[exp].to_dict()
            best_model_ = [[[i, best_model_[i]],["{}".format(i), best_model_["{}_condition".format(i)]]]  for i in variables]

        svr = PlotSurvival(data, time=time, censor=censor)

        _expression = []
        for vx, [[var, vqt], [var_condition, vcd]] in enumerate( best_model_ ):

            conditional_value = np.quantile(np.array(data[data[arm] == treatment][var]), vqt/100)
            val_condition = "(data['{}'] {} {}) ".format(var, vcd, conditional_value)
            _expression.append(val_condition)
        
        ix = eval( '&'.join(_expression) )
        
        svr.add(ix & (data[arm] == treatment), '{} (H)'.format(treatment))
        svr.add(~ix & (data[arm] == treatment), '{} (L)'.format(treatment))

        ref_ = '{} (L)'.format(treatment)
        tar = '{} (H)'.format(treatment)
        svr.plot([], ref = ref_, targets=[ref_, tar], colors=['black', '#5151bd', 'red'], line_styles=['-', '-', '-'], table=False, plot=False, legend=False)

        perf_ = {
                    "Reference": ref_, 
                    'Target': tar, 
                    'HR': svr.kmfs[tar][3][0], 
                    'Pvalue': svr.kmfs[tar][3][1], 
                    'CI_Low': svr.kmfs[tar][3][2], 
                    'CI_High': svr.kmfs[tar][3][3], 
                    'Reference_population': Counter(ix)[True],
                    'Target_population': Counter(ix)[False],
                }

        perf_.update({"percentile_{}".format(k): np.array(h[:1][k])[0] for k in feat})
        perf_.update({"{}_cutoff".format(k): np.percentile(test[k], np.array(h[:1][k])[0]) for k in feat})
    
        if not conditions:
            perf_.update({"{}_condition".format(k): np.array(h[:1]["{}_condition".format(k)])[0] for k in feat})
        

        perf.append(perf_)
    
    pickle.dump(
        {"Performance": perf, "Training": H},
        open('{}.pk'.format(outfile), 'wb')
    )


    return perf, H