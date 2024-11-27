# Clase con todas mis funciones de preprocesamiento de datos
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformation(BaseEstimator,TransformerMixin):

    def __init__(self, variables):
        
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        
        self.variables = variables

    def fit(self, X, y = None):

        return self
    
    def transform(self, X):

        for feature in self.variables:
            if (X[feature] <= 0).any():
                raise ValueError(f"Log transformation cannot be applied to non-positive values in {feature}")
            X[feature] = np.log(X[feature])
        
        return X
    
class Scaler(BaseEstimator,TransformerMixin):

    def __init__(self, variables):
        
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        
        self.variables = variables

    def fit(self, X, y = None):
        
        self.scaling_params_ = {
            feature: {'min': X[feature].min(), 'max': X[feature].max()}
            for feature in self.variables
        }

        return self
    
    def transform(self, X):

        for feature in self.variables:
            params = self.scaling_params_[feature]
            X[feature] = (X[feature] - params['min']) / (params['max'] - params['min'])
            
        return X