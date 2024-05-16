# Utils
import torch
import numpy as np

# Explanation Models
from openxai.explainers import Gradient
from openxai.explainers import IntegratedGradients
from openxai.explainers import InputTimesGradient
from openxai.explainers import SmoothGrad
from openxai.explainers import LIME
from openxai.explainers import SHAPExplainerC
from openxai.explainers import RandomBaseline


def Explainer(method: str,
              model,
              dataset_tensor: torch.tensor,
              param_dict=None):
    
    if method == 'grad':
        if param_dict is None:
            param_dict = dict()
            param_dict['absolute_value'] = True
        explainer = Gradient(model,
                             absolute_value=param_dict['absolute_value'])
    
    elif method == 'sg':
        if param_dict is None:
            param_dict = dict()
            param_dict['n_samples'] = 100
            param_dict['standard_deviation'] = 0.005
        explainer = SmoothGrad(model,
                               num_samples=param_dict['n_samples'],
                               standard_deviation=param_dict['standard_deviation'])
    
    elif method == 'itg':
        explainer = InputTimesGradient(model)
    
    elif method == 'ig':
        if param_dict is None:
            param_dict = dict()
            param_dict['method'] = 'gausslegendre'
            param_dict['multiply_by_inputs'] = False
            param_dict['baseline'] = torch.mean(dataset_tensor, dim=0).reshape(1, -1).float()
        explainer = IntegratedGradients(model,
                                        method=param_dict['method'],
                                        multiply_by_inputs=param_dict['multiply_by_inputs'],
                                        baseline=param_dict['baseline'],
                                        n_steps=param_dict['n_steps'])
    
    elif method == 'shap':
        if param_dict is None:
            param_dict = dict()
            param_dict['subset_size'] = 500
        explainer = SHAPExplainerC(model,
                                   model_impl='torch',
                                   n_samples=param_dict['n_samples'])

    elif method == 'lime':
        if param_dict is None:
            param_dict = dict()
            param_dict['dataset_tensor'] = dataset_tensor
            param_dict['kernel_width'] = 0.75
            param_dict['std'] = float(np.sqrt(0.05))
            param_dict['mode'] = 'tabular'
            param_dict['sample_around_instance'] = True
            param_dict['n_samples'] = 1000
            param_dict['discretize_continuous'] = False
            param_dict['categorical_features'] = []  # changed by Nick K

        explainer = LIME(model,#.predict,
                         param_dict['dataset_tensor'],
                         std=param_dict['std'],
                         mode=param_dict['mode'],
                         sample_around_instance=param_dict['sample_around_instance'],
                         kernel_width=param_dict['kernel_width'],
                         n_samples=param_dict['n_samples'],
                         discretize_continuous=param_dict['discretize_continuous'],
                         categorical_features=param_dict['categorical_features'])  # changed by Nick K

    elif method == 'control':
        explainer = RandomBaseline(model)
    
    else:
        raise NotImplementedError("This method has not been implemented, yet.")
    
    return explainer
