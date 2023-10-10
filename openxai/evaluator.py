import numpy as np
import torch
from scipy.stats import rankdata
import pandas as pd

class Evaluator():
    """ Metrics to evaluate an explanation method."""
    def __init__(self, input_dict: dict):
        self.input_dict = input_dict
        self.model = input_dict['model']
        if hasattr(self.model, 'return_ground_truth_importance'):
            self.gt_feature_importances = self.model.return_ground_truth_importance()
        self.explanation_x_f = self.input_dict['explanation_x']

    def evaluate(self, metric: str):
        """Explanation evaluation of a given metric."""
        if not hasattr(self.model, 'return_ground_truth_importance') and metric in ['FA', 'RA']:
            raise ValueError("This chosen metric is incompatible with non-linear models.")

        # Feature Agreement
        if metric == 'FA':
            scores, average_score = self.agreement_fraction(metric='overlap')
            return scores, average_score
        # Rank Agreement
        elif metric == 'RA':
            scores, average_score = self.agreement_fraction(metric='rank')
            return scores, average_score
        # Prediction Gap on Important Features
        elif metric == 'PGI':
            scores = self.eval_pred_faithfulness(num_samples=self.input_dict['perturb_num_samples'], invert=False)
            return scores
        # Prediction Gap on Unimportant Features
        elif metric == 'PGU':
            scores = self.eval_pred_faithfulness(num_samples=self.input_dict['perturb_num_samples'], invert=True)
            return scores
        else:
            raise NotImplementedError("This metric is not implemented in this OpenXAI version.")

    def agreement_fraction(self, metric):
        """FA and RA calcuation."""
        attrA = self.gt_feature_importances.detach().numpy().reshape(1, -1)
        attrB = self.explanation_x_f.detach().numpy().reshape(1, -1)
        k = self.input_dict['top_k']

        # id of top-k features
        topk_idA = np.argsort(-np.abs(attrA), axis=1)[:, 0:k]
        topk_idB = np.argsort(-np.abs(attrB), axis=1)[:, 0:k]

        # rank of top-k features --> manually calculate rankings (instead of using 0, 1, ..., k ranking based on argsort output) to account for ties
        all_feat_ranksA = rankdata(-np.abs(attrA), method='dense', axis=1) #rankdata gives rank1 for smallest # --> we want rank1 for largest # (aka # with largest magnitude)
        all_feat_ranksB = rankdata(-np.abs(attrB), method='dense', axis=1)
        topk_ranksA = np.take_along_axis(all_feat_ranksA, topk_idA, axis=1)
        topk_ranksB = np.take_along_axis(all_feat_ranksB, topk_idB, axis=1)

        # overlap agreement = (# topk features in common)/k
        if metric == 'overlap':
            topk_setsA = [set(row) for row in topk_idA]
            topk_setsB = [set(row) for row in topk_idB]
            # check if: same id
            metric_distr = np.array([len(setA.intersection(setB))/k for setA, setB in zip(topk_setsA, topk_setsB)])

        # rank agreement
        elif metric == 'rank':
            topk_idA_df = pd.DataFrame(topk_idA).applymap(str)  # id
            topk_idB_df = pd.DataFrame(topk_idB).applymap(str)
            topk_ranksA_df = pd.DataFrame(topk_ranksA).applymap(str)  # rank (accounting for ties)
            topk_ranksB_df = pd.DataFrame(topk_ranksB).applymap(str)
            
            #check if: same id + rank
            topk_id_ranksA_df = ('feat' + topk_idA_df) + ('rank' + topk_ranksA_df)
            topk_id_ranksB_df = ('feat' + topk_idB_df) + ('rank' + topk_ranksB_df)
            metric_distr = (topk_id_ranksA_df == topk_id_ranksB_df).sum(axis=1).to_numpy()/k

        else:
            raise NotImplementedError("Please make sure that have chosen one of the following metrics: {overlap (FA), rank (RA)}.")

        return metric_distr, np.mean(metric_distr)

    def _arr(self, x) -> np.ndarray:
        """ Converts x to a numpy array.
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)
            
    def eval_pred_faithfulness(self, num_samples: int = 100, invert: bool = False):
        """ Approximates the expected local faithfulness of the explanation
            in a neighborhood around input x.
        Args:
            num_perturbations: number of perturbations used for Monte Carlo expectation estimate
        """
        self._parse_and_check_input()

        if invert:
            self.top_k_mask = torch.logical_not(self.top_k_mask)
        
        # get perturbations of instance x
        x_perturbed = self.perturb_method.get_perturbed_inputs(original_sample=self.x,
                                                               feature_mask=self.top_k_mask,
                                                               num_samples=num_samples,
                                                               feature_metadata=self.feature_metadata)

        # Average the expected absolute difference.
        y = self._arr(self.model(self.x.reshape(1, -1).float()))
        y_perturbed = self._arr(self.model(x_perturbed.float()))

        return np.mean(np.abs(y - y_perturbed), axis=0)[0]

    def _parse_and_check_input(self):
        """PGI AND PGU checks."""
        if not 'x' in self.input_dict:
            raise ValueError('Missing key of x')
        self.x = self.input_dict['x']

        # predictive model
        if not 'model' in self.input_dict:
            raise ValueError('Missing key of model')
        self.model = self.input_dict['model']

        # top-K parameter K
        if not 'top_k' in self.input_dict:
            raise ValueError('Missing key of top_k')
        self.top_k = self.input_dict['top_k']

        # initialized perturbation method object BasePerturbation pertub_method
        if not 'perturb_method' in self.input_dict:
            raise ValueError('Missing key of perturbation method BasePerturbation perturb_method')
        # initialize the perturbation method, which extends from BasePerturbation
        self.perturb_method = self.input_dict['perturb_method']

        # initialized perturbation method object BasePerturbation pertub_method
        if not 'feature_metadata' in self.input_dict:
            raise ValueError('Missing key of feature metadata feature_metadata')
        # initialize the perturbation method, which extends from BasePerturbation
        self.feature_metadata = self.input_dict['feature_metadata']
        self.top_k_mask = self.input_dict['mask']
