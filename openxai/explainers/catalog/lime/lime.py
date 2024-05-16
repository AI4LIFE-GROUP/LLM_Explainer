import numpy as np
import torch
from tqdm import tqdm
from ... api import Explainer

# import lime
from .lime_package import lime_tabular
from .lime_package import lime_image
from .lime_package import lime_text


class LIME(Explainer):
    """
    This class gets explanations for tabular data. The explanations are generated according to Ribeiro et al's tabular
    sampling algorithm.

    model : model object
    data : np array
    mode : str, "tabular" or "images"
    """

    def __init__(self, model, data: torch.FloatTensor, std, mode: str = "tabular", sample_around_instance: bool = True,
                 kernel_width: float = 0.75, n_samples: int = 1000, discretize_continuous: bool = False,
                 categorical_features = None) -> None:  # changed by Nick K

        self.output_dim = 2
        self.mode = mode
        self.model = model
        self.n_samples = n_samples
        self.discretize_continuous = discretize_continuous
        self.sample_around_instance = sample_around_instance
        if categorical_features is None:  # changed by Nick K
            categorical_features = []

        if self.mode == "tabular":
            self.data = data.numpy()
            self.explainer = lime_tabular.LimeTabularExplainer(self.data,
                                                               mode="classification",
                                                               sample_around_instance=self.sample_around_instance,
                                                               discretize_continuous=self.discretize_continuous,
                                                               kernel_width=kernel_width * np.sqrt(
                                                                        self.data.shape[1]),
                                                               std=std,
                                                               categorical_features=categorical_features  # changed by Nick K
                                                               )
        elif self.mode == 'text':
            self.explainer = lime_text.LimeTextExplainer(class_names=['0', '1'])
        else:
            self.explainer = lime_image.LimeImageExplainer()

        super(LIME, self).__init__(model)

    def get_explanation(self, all_data: torch.FloatTensor, seed=None, perturbations=None, disable_tqdm=False) -> torch.FloatTensor:

        if self.mode == "tabular":
            all_data = all_data.numpy()
            num_features = all_data.shape[1]
            attribution_scores = [np.zeros(all_data.shape) for j in range(self.output_dim)]

            all_perturbations = []
            for i in tqdm(range(all_data.shape[0]), disable=disable_tqdm):
                exp, perts = self.explainer.explain_instance(all_data[i, :], self.model,
                                                                     num_features=num_features,
                                                                     num_samples=self.n_samples,
                                                                     seed=seed, perturbations=perturbations)

                # bring explanations into data order (since LIME automatically orders according to highest importance)
                for j in range(self.output_dim):
                    for k, v in exp.local_exp[1]:
                        attribution_scores[j][i, k] = v
                all_perturbations.append(perts)
            # we are explaining the the prob of 1; choosing [0] would explain P(y=0|x)
            return torch.FloatTensor(attribution_scores[1]), all_perturbations
        elif self.mode == 'text':
            attribution_scores = []
            for i in tqdm(range(all_data.shape[0]), disable=disable_tqdm):
                text = all_data[i]
                exp = self.explainer.explain_instance(text, self.model, num_samples=self.n_samples)
                attribution_scores.append(exp)

            return attribution_scores
        else:
            attribution_scores = []
            for i in tqdm(range(all_data.shape[0]), disable=disable_tqdm):
                img = all_data  # .detach().numpy()
                # img = np.transpose(img, (1, 2, 0)).astype('double')

                # lime requires an image input size of (height, width, channels)
                # the classification wrapper within the cnn method needs to be adjusted accordingly
                # in this style: https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb

                exp = self.explainer.explain_instance(img,
                                                      self.model,
                                                      top_labels=5,
                                                      hide_color=0,
                                                      num_samples=self.n_samples)
                attribution_scores.append(exp)

            return torch.FloatTensor(attribution_scores)
