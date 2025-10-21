import numpy as np 
import pandas as pd

from lifelines import KaplanMeierFitter

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

class DataLoader():
    '''
    Generate dataset for training
    
    dataloader = PLFDataset(
        features,
        time='OS_Months',
        event='OS_Event',
        treatment='Treatment',
        normalize=True
    )

    X_train, y_train = dataloader.fit(data_train)
    X_test, y_test = dataloader.transform(data_train)
    '''
    
    
    def __init__(self, features=[], time='time', event='event', treatment=None, normalize=True, **kwargs):
        '''This class assumes that the data is already in good shape and with no categorical variables'''

        self.time=time
        self.event=event
        self.normalize=normalize
        self.features = features
        
    
    def fit(self, data):
        
        assert isinstance(data, pd.DataFrame)==True
        
        X = data[self.features].copy()
        y = data[[self.time, self.event]].copy()

        if self.normalize:
            self.xmin = X.min()
            self.xmax = X.max()
            X=(X-self.xmin)/(self.xmax-self.xmin)
        
        X=X.fillna(0).to_numpy() #convert to numpy for reshaping
        
        # lambdas=lambdas_(y[self.time], y[self.event]) #compute lamdas (see loss function formula for explanation)
        # y[self.lambdas]=lambdas
        y = y[[self.time, self.event]]
        
        return X, y
    
    def transform(self, data):
        assert isinstance(data, pd.DataFrame)==True
        
        X = data[self.features].copy()

        if self.normalize:
            X=(X-self.xmin)/(self.xmax-self.xmin)
        
        X=X.fillna(0).to_numpy() #convert to numpy for reshaping

        try:
            y = data[[self.time, self.event]].copy()        
            # lambdas=lambdas_(y[self.time], y[self.event]) #compute lamdas (see loss function formula for explanation)
            # y[self.lambdas]=lambdas
            y = y[[self.time, self.event]]
            
            
            return X, y
        except:
            return X, None