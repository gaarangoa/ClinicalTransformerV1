#!/usr/bin/env python
# coding: utf-8



import numpy as np
import tensorflow as tf


get_ipython().run_line_magic('matplotlib', 'inline')


import matplotlib
import matplotlib.pyplot as plt
from collections import Counter


import lifelines
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter


# In[2]:


import pandas as pd

def lambdas_(time, event):
    """
    lambda_i= Sum_t Omega_t / N_t * I(T_i>t)
    y is survival time with shape (n, )
    """
    kmf = KaplanMeierFitter()
    kmf.fit(durations=time, event_observed=event ) #fit km to your data

    #extract event table
    e_table=pd.DataFrame(kmf.event_table).reset_index() # reset index so that time is part of the data. 
    omegat_over_Nt=e_table["observed"]/e_table["at_risk"] # Omege_t/N_t term in the equation. 
    
    lamdas=np.zeros(time.shape[0]) #where to save lambda values 
    for ix, Ti in enumerate(time):

        lamdas[ix]=np.sum(omegat_over_Nt* ( Ti>e_table["event_at"]))
    return lamdas


#generate datagenerator for predictive biomarker loss function
class PredictiveLossDataLoader:
    '''
    data_x, predictor variables 
    time, OS
    event, OS event
    treatment, treatmeant information 
 
    
    '''
    
    
    def __init__(self, data_x,time, event, treatment, normalize=True, transformer=True):
        self.data_x=data_x
        self.time=time
        self.event=event
        self.treatment=treatment
        self.normalize=normalize
        self.transformer=transformer
        
    
    def datagenerator(self,):

        assert isinstance(self.data_x, pd.DataFrame)==True

        features=list(self.data_x)

        if self.normalize:
            data_x=(self.data_x-self.data_x.min())/(self.data_x.max()-self.data_x.min())
        else:
            data_x=self.data_x
            
        lambdas=lambdas_(self.time, self.event) #compute lamdas (see loss function formula for explanation)

        data_x=data_x.to_numpy() #convert to numpy for reshaping
        
        if self.transformer:
            data_x=data_x.reshape([data_x.shape[0], 1, len(features)]) #reshape to dimensions expected by transformer.

        data_y=np.zeros( (self.time.shape[0], 4)) #[time, event, lambdas, treatment]
        data_y[:,0]=self.time
        data_y[:,1]=self.event
        data_y[:,2]=lambdas
        data_y[:,3]=self.treatment
        
        return data_x, data_y

    
#plot predictive loss Kaplan meier
def predlosskmplot(score, ydata, cutoff=0, use_predscore=True, figsize=(15,5), fontsize=9,labelsize=10,overalltitle='', 
                  treatmentA='Treatment A', treatmentB='Treatment B',varname='var', ms=6):
    
    kmf_BP_TP = KaplanMeierFitter()
    kmf_BP_TN = KaplanMeierFitter()
    
    cph=CoxPHFitter()

    #from lifelines.plotting import add_at_risk_counts
    
    treatment = ydata[:,3] ==1
    T = ydata[:,0]
    E = ydata[:,1]  

    #btmb>btmb_cutoff
    if use_predscore:
        
        biomarker_finder =score.squeeze() <0
        
    else:
    
        biomarker_finder =score>cutoff
    
    #take of situations when the all predictions are >0 or <0 
    if len(Counter(biomarker_finder))==2:
        
        fig, ax = plt.subplots(1,2, figsize=figsize)

        #compute hazard ratio
        mydata=pd.DataFrame(ydata[:,[0,1,3]], columns=['os', 'os_event','treatment'])[biomarker_finder]
        cph.fit(mydata, duration_col='os', event_col='os_event')
        hr1=np.round(cph.hazard_ratios_[0], 3)

        kmf = KaplanMeierFitter()
        kmf1 = KaplanMeierFitter()

        T_Bp_Tp = T[biomarker_finder & treatment]
        T_Bp_Tn = T[biomarker_finder & ~treatment]

        E_Bp_Tp = E[biomarker_finder & treatment]
        E_Bp_Tn = E[biomarker_finder & ~treatment]

        kmf.fit(T_Bp_Tp, event_observed=E_Bp_Tp, label=treatmentA+" (" + str(len(T_Bp_Tp)) + ")")
        kmf.plot(ax=ax[0],ci_show=False, 
                 show_censors=True,
                       censor_styles={'ms':ms, 'marker': '+'},
                  linewidth=1, color='b')


        kmf1.fit(T_Bp_Tn, event_observed=E_Bp_Tn, label=treatmentB+ " (" + str(len(T_Bp_Tn)) + ")")
        kmf1.plot(ax=ax[0],ci_show=False, 
                  show_censors=True,
                       censor_styles={'ms':ms, 'marker': '+'},
                  linewidth=1, color="k", linestyle='-')

        #add median information 
        ax[1].set_ylim(0.0,1.1)
        ax[0].set_ylim(0.0,1.1)

        ax[0].hlines(0.5, ax[0].get_xlim()[0], 
                     np.max([ kmf.median_survival_time_, kmf1.median_survival_time_]), linestyle='--', color='y', lw=2)


        ax[0].vlines(kmf.median_survival_time_, ax[0].get_ylim()[0], 
                     0.5, linestyle='--',
                     color='y', lw=2)


        ax[0].vlines(kmf1.median_survival_time_, ax[0].get_ylim()[0], 
                     0.5, linestyle='--', color='y', lw=2)

        if np.abs(kmf.median_survival_time_-kmf1.median_survival_time_)<=4:
            d1=0.1
            d2=0.05
        else:
            d1=0.1
            d2=0.1


        ax[0].text(kmf.median_survival_time_, d1,
                   str(np.round( kmf.median_survival_time_, 1) ),fontsize=fontsize,fontweight='bold')

        ax[0].text(kmf1.median_survival_time_, d2,
                   str(np.round( kmf1.median_survival_time_, 1) ),fontsize=fontsize, fontweight='bold')

        ax[0].text(0.1, 0.1,
                   'HR : '+str(hr1), fontsize=fontsize,fontweight='normal')

        ax[0].legend(fontsize=fontsize)  #.set_visible(False)

        if use_predscore:
            ax[0].set_title("Group 1", fontsize=fontsize, fontweight='bold')
        else:
            ax[0].set_title(varname+" > "+str(cutoff), fontsize=fontsize, fontweight='bold')

        ax[0].set_ylabel("Survival Probabilities", fontsize=fontsize, fontweight='bold')
        #ax[0].set_xlabel('', fontsize=fontsize, fontweight='bold')

        ax[0].set_xlabel("Time", fontsize=fontsize, fontweight='bold')
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].xaxis.set_tick_params(labelsize=labelsize)
        ax[0].yaxis.set_tick_params(labelsize=labelsize)
        ax[0].set_xlim(0.0,ax[0].get_xlim()[1])


         #btmb<btmb_cutoff
        #compute hazard ratio
        mydata1=pd.DataFrame(ydata[:,[0,1,3]], columns=['os', 'os_event','treatment'])[~biomarker_finder]
        cph.fit(mydata1, duration_col='os', event_col='os_event')
        hr2=np.round(cph.hazard_ratios_[0], 3)


        kmf = KaplanMeierFitter()
        kmf1 = KaplanMeierFitter()

        T_Bn_Tp = T[~biomarker_finder & treatment]
        T_Bn_Tn = T[~biomarker_finder & ~treatment]

        E_Bn_Tp = E[~biomarker_finder & treatment]
        E_Bn_Tn = E[~biomarker_finder & ~treatment]

        kmf.fit(T_Bn_Tp, event_observed=E_Bn_Tp, label=treatmentA+ " (" + str(len(T_Bn_Tp)) + ")")
        kmf.plot(ax=ax[1],ci_show=False,
                 show_censors=True,
                       censor_styles={'ms':ms, 'marker': '+'},
                  linewidth=1, color='b')

        kmf1.fit(T_Bn_Tn, event_observed=E_Bn_Tn, label=treatmentB + " (" + str(len(T_Bn_Tn)) + ")")
        kmf1.plot(ax=ax[1], ci_show=False,
                  show_censors=True,
                       censor_styles={'ms':ms, 'marker': '+'},
                  linewidth=1, color='k',linestyle='-')

        #add median information 
        ax[1].hlines(0.5, ax[1].get_xlim()[0], 
                     np.max([ kmf.median_survival_time_, kmf1.median_survival_time_]),
                     linestyle='--', color='y', lw=2)

        ax[1].vlines(kmf1.median_survival_time_, ax[1].get_ylim()[0], 
                     0.5, linestyle='--', color='y', lw=2)

        ax[1].vlines(kmf.median_survival_time_, ax[1].get_ylim()[0], 
                     0.5, linestyle='--', color='y', lw=2)

        if np.abs(kmf.median_survival_time_-kmf1.median_survival_time_)<=4:
            d1=0.1
            d2=0.05
        else:
            d1=0.1
            d2=0.1


        ax[1].text(kmf.median_survival_time_, d1,
                   str(np.round( kmf.median_survival_time_, 1) ), fontsize=fontsize,fontweight='bold')


        ax[1].text(kmf1.median_survival_time_, d2,
                   str(np.round( kmf1.median_survival_time_, 1) ), fontsize=fontsize,fontweight='bold')


        ax[1].text(0.1, 0.1,
                   'HR : '+str(hr2), fontsize=fontsize,fontweight='normal')


        ax[1].legend(fontsize=fontsize) #.set_visible(False)

        if use_predscore:
            ax[1].set_title("Group 2", fontsize=fontsize, fontweight='bold')
        else:
            ax[1].set_title(varname+" <= "+str(cutoff), fontsize=fontsize, fontweight='bold')

        ax[1].set_ylabel("Survival Probabilities", fontsize=fontsize, fontweight='bold')
        #ax[1].set_xlabel('', fontsize=fontsize, fontweight='bold')

        ax[1].set_xlabel("Time", fontsize=fontsize, fontweight='bold')
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].xaxis.set_tick_params(labelsize=labelsize)
        ax[1].yaxis.set_tick_params(labelsize=labelsize)

        ax[1].set_xlim(0.0,ax[1].get_xlim()[1])
        fig.suptitle(overalltitle, fontsize=16, fontweight='bold')
        
        #take care of situations where all prediction > 0 or <0
    else:
        
        fig, ax = plt.subplots(1,1, figsize=(7,5))
        #compute hazard ratio
        mydata=pd.DataFrame(ydata[:,[0,1,3]], columns=['os', 'os_event','treatment'])
        cph.fit(mydata, duration_col='os', event_col='os_event')
        hr1=np.round(cph.hazard_ratios_[0], 3)

        kmf = KaplanMeierFitter()
        kmf1 = KaplanMeierFitter()

        T_Bp_Tp = T[treatment]
        T_Bp_Tn = T[~treatment]

        E_Bp_Tp = E[treatment]
        E_Bp_Tn = E[~treatment]

        kmf.fit(T_Bp_Tp, event_observed=E_Bp_Tp, label=treatmentA+ " (" + str(len(T_Bp_Tp)) + ")")
        kmf.plot(ax=ax,ci_show=False, 
                 show_censors=True,
                       censor_styles={'ms':ms, 'marker': '+'},
                  linewidth=1, color='b')


        kmf1.fit(T_Bp_Tn, event_observed=E_Bp_Tn, label=treatmentA+ " (" + str(len(T_Bp_Tn)) + ")")
        kmf1.plot(ax=ax,ci_show=False, 
                  show_censors=True,
                       censor_styles={'ms':ms, 'marker': '+'},
                  linewidth=1, color="k", linestyle='-')

        #add median information 
        ax.set_ylim(0.0,1.1)
        ax.set_ylim(0.0,1.1)

        ax.hlines(0.5, ax.get_xlim()[0], 
                     np.max([ kmf.median_survival_time_, kmf1.median_survival_time_]), linestyle='--', color='y', lw=2)


        ax.vlines(kmf.median_survival_time_, ax.get_ylim()[0], 
                     0.5, linestyle='--',
                     color='y', lw=2)


        ax.vlines(kmf1.median_survival_time_, ax.get_ylim()[0], 
                     0.5, linestyle='--', color='y', lw=2)

        if np.abs(kmf.median_survival_time_-kmf1.median_survival_time_)<=4:
            d1=0.1
            d2=0.05
        else:
            d1=0.1
            d2=0.1


        ax.text(kmf.median_survival_time_, d1,
                   str(np.round( kmf.median_survival_time_, 1) ),fontsize=fontsize,fontweight='bold')

        ax.text(kmf1.median_survival_time_, d2,
                   str(np.round( kmf1.median_survival_time_, 1) ),fontsize=fontsize, fontweight='bold')

        ax.text(0.1, 0.1,
                   'HR : '+str(hr1), fontsize=fontsize,fontweight='normal')

        ax.legend(fontsize=fontsize)  #.set_visible(False)

        if use_predscore:
            ax.set_title("Variable not predictive\nAll patients assigned to one group", fontsize=fontsize, fontweight='normal')
        else:
            ax.set_title(varname, fontsize=fontsize, fontweight='bold')

        ax.set_ylabel("Survival Probabilities", fontsize=fontsize, fontweight='bold')

        ax.set_xlabel("Time", fontsize=fontsize, fontweight='bold')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_tick_params(labelsize=labelsize)
        ax.yaxis.set_tick_params(labelsize=labelsize)
        ax.set_xlim(0.0,ax.get_xlim()[1])
