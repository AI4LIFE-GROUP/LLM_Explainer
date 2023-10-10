import torch
import numpy as np
from LLM_Explainer.openxai.explainers.catalog.perturbation_methods import NormalPerturbation


class Sampler:
    def __init__(self, model, sample_space):
        self.model = model
        self.sample_space_X = sample_space

    def sample(self, n_shot, seed, use_soft_preds, use_most_confident,
               use_class_balancing, sorting='alternate'):
        # Generate perturbations/nearest neighbours if necessary
        self.update_sample_space()

        # Get soft and hard predictions
        y = self.model(self.sample_space.float())
        soft_preds = y[:, 1].detach().numpy()
        hard_preds = y.argmax(axis=-1).detach().numpy()

        # Get ICL samples
        ICL_idxs = self.get_ICL_indices(soft_preds, n_shot, seed, use_most_confident,
                                        use_class_balancing, sorting)
        X_ICL = self.sample_space[ICL_idxs]
        y_ICL = soft_preds[ICL_idxs] if use_soft_preds else hard_preds[ICL_idxs]

        return X_ICL.detach().numpy(), y_ICL

    def get_ICL_indices(self, soft_preds, n_shot, seed, use_most_confident, use_class_balancing, sorting='alternate'):
        y_neg = np.where(soft_preds < 0.5)[0]
        y_pos = np.where(soft_preds >= 0.5)[0]

        if use_class_balancing:
            if len(y_neg) < (n_shot // 2) or len(y_pos) < (n_shot // 2):
                print("You don't have enough perturbations for a balanced ICL prompt!")
                print("You have {} negative perturbations and {} positive perturbations".format(len(y_neg), len(y_pos)))

        # Sort indices by confidence
        sorted_indices = np.argsort(soft_preds)

        # Get indices
        if use_most_confident and use_class_balancing:
            # Get n_shot//2 instances from each end of the distribution
            y_neg_idx = sorted_indices[:n_shot // 2]
            y_pos_idx = sorted_indices[-n_shot // 2:]
            # shuffle indices with seed
            np.random.seed(seed)
            np.random.shuffle(y_neg_idx)
            np.random.seed(seed)
            np.random.shuffle(y_pos_idx)

            # interleave indices (sorting='alternate') or shuffle (sorting='shuffle')
            ICL_idxs = self._sort_icl_idxs(y_neg_idx, y_pos_idx, sorting, seed)
        elif use_most_confident and not use_class_balancing:
            raise NotImplementedError("use_most_confident=True and use_class_balancing=False not implemented")
        elif not use_most_confident and use_class_balancing:
            # randomly sample from negative and positive instances separately
            np.random.seed(seed)
            if len(y_neg) < (n_shot // 2):
                # Get all from minority class, get remainder random
                y_neg_idx = y_neg
                y_pos_idx = np.random.choice(y_pos, n_shot - len(y_neg), replace=False)
            elif len(y_pos) < (n_shot // 2):
                # Get all from majority class, get remainder random
                y_pos_idx = y_pos
                y_neg_idx = np.random.choice(y_neg, n_shot - len(y_pos), replace=False)
            else:
                # Get half from each class
                y_neg_idx = np.random.choice(y_neg, n_shot // 2, replace=False)
                y_pos_idx = np.random.choice(y_pos, n_shot // 2, replace=False)

            # interleave indices (sorting='alternating') or shuffle (sorting='shuffled')
            ICL_idxs = self._sort_icl_idxs(y_neg_idx, y_pos_idx, sorting, seed)
        elif not use_most_confident and not use_class_balancing:
            # randomly sample from all instances
            np.random.seed(seed)
            ICL_idxs = np.random.choice(np.arange(len(soft_preds)), n_shot, replace=False)

        return ICL_idxs
    
    def _sort_icl_idxs(self, y_neg_idx, y_pos_idx, sorting, seed=None):
        if sorting == 'alternate':
            return self._interleave_indices(y_neg_idx, y_pos_idx)
        elif sorting == 'shuffle':
            np.random.seed(seed)
            # concatenate y_neg_idx and y_pos_idx, shuffle, and return
            return np.random.permutation(np.concatenate([y_neg_idx, y_pos_idx]))
        else:
            raise ValueError("sorting must be 'alternate' or 'shuffle'")
    
    def _interleave_indices(self, y_neg_idx, y_pos_idx):
        n_shot = len(y_neg_idx) + len(y_pos_idx)
        ICL_idxs = []
        i, j = 0, 0

        # Loop through the indices and interleave
        while len(ICL_idxs) < n_shot:
            if i < len(y_neg_idx):
                ICL_idxs.append(y_neg_idx[i])
                i += 1
                
            if j < len(y_pos_idx):
                ICL_idxs.append(y_pos_idx[j])
                j += 1
                
        return ICL_idxs
        
    def update_sample_space(self):
        raise NotImplementedError("update_sample_space() must be implemented by child class")

class ConstantICL(Sampler):
    def __init__(self, model, sample_space):
        super().__init__(model, sample_space)

    def update_sample_space(self):
        return

class PerturbICL(Sampler):
    def __init__(self, model, sample_space, X_test,
                 feature_types, **perturb_params):
        super().__init__(model, sample_space)  # Base class init

        # Init variables
        self.model = model
        self.X_test = X_test
        self.feature_types = feature_types
        self.std = perturb_params['std']
        self.n_samples = perturb_params['n_samples']
        self.perturb_seed = perturb_params['perturb_seed']

        # Compute variables
        self.num_features = self.X_test.shape[-1]
        self.feature_mask = np.zeros(self.num_features, dtype=bool)

    def update_sample_space(self):
        flip = np.sqrt(2/np.pi)*self.std
        perturbation = NormalPerturbation("tabular", mean=0, std_dev=self.std,
                                          flip_percentage=flip, seed=self.perturb_seed)
        self.sample_space = perturbation.get_perturbed_inputs(self.X_test[self.eval_idx],
                                                              self.feature_mask,
                                                              self.n_samples,
                                                              self.feature_types)#.detach().numpy()

# def most_confident_preds(model, perturbations, eval_idx, n_shot):
#     perturbations_i = np.array(perturbations[eval_idx])
#     preds = model.predict(torch.tensor(perturbations_i).float()).argmax(axis=1)
#
#     # subset N//2 0s and N//2 1s
#     y_neg = np.where(preds == 0)[0]
#     y_pos = np.where(preds == 1)[0]
#     assert len(y_neg) >= (n_shot // 2) and len(y_pos) >= (
#                 n_shot // 2), "You don't have enough perturbations for a balanced ICL prompt! " \
#                               "Use more n_samples for LIME perturbations."
#
#     # get most confident samples and indices for neg and pos classes
#     most_conf_samples_neg, most_conf_inds_neg = get_high_confidence_samples(
#         torch.tensor(perturbations_i[y_neg]).float(), model, n_shot // 2)
#     most_conf_samples_pos, most_conf_inds_pos = get_high_confidence_samples(
#         torch.tensor(perturbations_i[y_pos]).float(), model, n_shot // 2)
#
#     # concat neg and pos most confident predictions
#     X_ICL = [None] * n_shot
#     X_ICL[::2] = most_conf_samples_neg.numpy()
#     X_ICL[1::2] = most_conf_samples_pos.numpy()
#     y_ICL = [None] * n_shot
#     y_ICL[::2] = preds[y_neg][most_conf_inds_neg]
#     y_ICL[1::2] = preds[y_pos][most_conf_inds_pos]
#     X_ICL, y_ICL = np.array(X_ICL), np.array(y_ICL)
#     return X_ICL, y_ICL

def compute_confidence(outputs):
    """
    outputs : torch.tensor : vector with probability scores assigned to each class.
    """
    std_vals = torch.std(outputs, dim = 1)
    return std_vals

def top_k_indices(tensor, k):
    _, indices = torch.topk(tensor, k)
    return indices

def bottom_k_indices(tensor, k):
    _, indices = torch.topk(tensor, k, largest=False)
    return indices

def get_high_confidence_samples(val_samples, model, k):
    """
    model : Pytorch Model
    val_samples = Torch.tensor of N X D dimension
    """
    outputs = model(val_samples)
    confidence_scores = compute_confidence(outputs)
    top_indices = top_k_indices(confidence_scores, k)
    return val_samples[top_indices], top_indices

def get_low_confidence_samples(val_samples, model, k):
    """
    model : Pytorch Model
    """
    outputs = model(val_samples)
    confidence_scores = compute_confidence(outputs)
    bottom_indices = bottom_k_indices(confidence_scores, k)
    return val_samples[bottom_indices]

