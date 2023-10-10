# LLM_Explainer 
Codebase for the paper [Are Large Language Models Post Hoc Explainers?](https://arxiv.org/abs/2310.05797)

![LLM_framework_pages-to-jpg-0001](https://github.com/AI4LIFE-GROUP/LLM_Explainer/assets/35569862/ecee3472-6537-4761-a489-ed1d2b5399a3)

# Contents 

The repository is organized as follows:

The ```data``` folder contains the pre-processed Blood, COMPAS, Credit and Adult datasets.

The ```llms``` folder contains code for prompt generation, LLM API calls, and response processing.

The ```models``` folder contains pre-trained Logistic Regression (LR) and Large Artificial Neural Network (ANN-L) classifiers.

The ```openxai``` folder contains code from [Agarwal et al. 2022](https://arxiv.org/abs/2206.11104), used to generate post-hoc explanations, perturbations around test samples, and faithfulness scores.

The ```notebooks``` folder contains demonstrations such as model training and model inspection.

The ```outputs``` folder stores results from a) post-hoc explainers and b) LLM explainers.

# Pipeline Instructions

### LLM Explanations

To generate explanations from a given LLM, run the following command:

```
python3 LLM_PostHocPipeline.py
```

The parameters used are located in the config file ```LLM_pipeline_config.json```.
- `data_name` &mdash; Name of the dataset to use: "blood", "compas", "credit" or "adult" (default &mdash; "adult")
- `data_scaler` &mdash; Scaler for the data: "minmax", "standard" or "none" (default &mdash; "minmax")
- `model_name` &mdash; Name of the model to use, e.g., "lr", "ann_l", etc. (default &mdash; "lr")
- `base_model_dir` &mdash; Directory of the saved model (default &mdash; "./models/ClassWeighted_scale_minmax/")
- `output_dir` &mdash; Directory to save LLM results to (default &mdash; "./outputs/LLM_QueryAndReply/")
- `openai_api_key_file_path` &mdash; File path to your OpenAI API key (default &mdash; "./openai_api_key.txt")
- `LLM_name` &mdash; Name of the LLM model (default &mdash; "gpt-4")
- `temperature` &mdash; Parameter controlling the randomness of the LLM's output (default &mdash; 0)
- `eval_min_idx` &mdash; The minimum test sample index for evaluation (default &mdash; 0)
- `eval_max_idx` &mdash; The maximum test sample index for evaluation (default &mdash; 100)
- `max_test_samples` &mdash; A hard limit on the number of test samples to evaluate (default &mdash; 100)
- `SEED` &mdash; Seed value for reproducibility (default &mdash; 0)
- `n_shot` &mdash; The number of examples used for in-context learning (ICL) the model (default &mdash; 16)
- `icl_params' &mdash; parameters for controlling the generation of ICL examples, see below
- `sampling_params` &mdash; Parameters for the different sampling strategies
- `prompt_params` &mdash; parameters for controlling the prompt of the model, see below
- `experiment_params` &mdash; parameters used to identify the experiment, see below

The prompts for querying the LLM are stored in the appropriate config file (default &mdash; ```prompts.json```). From within these file, users have the ability to control the following parameters:

### Additional Parameters

The `sampling_params` dictionary contains the following schemes:

- `constant_icl` &mdash; Empty dictionary
- `lime_sample` &mdash; Empty dictionary
- `most_confident_preds`&mdash; Empty dictionary

#### ICL Parameters

- `icl_seed` &mdash; The seed used to generate ICL samples (default &mdash; 0)
- `sampling_scheme` &mdash; ICL sampling strategy, see below (default &mdash; "most_confident_preds")

#### Prompt Parameters

The `prompt_params` dictionary contains the following parameters:

- `k` &mdash; The number of top-K features to request from the LLM. Use -1 for all features (default &mdash; -1)
- `hide_feature_details` &mdash; Boolean controlling whether or not feature names and suffixes (e.g., Age is 27 years vs A is 27) are hidden (default &mdash; true)
- `input_str` &mdash; String to prepend to each ICL input (default &mdash; "")
- `output_str` &mdash; String or list to prepend to each ICL output. For strings, use e.g., "Output &mdash; " (default &mdash; ["Output &mdash; 0.", "Output &mdash; 1."])
- `input_sep` &mdash; Separator between blocks of ICL inputs (default &mdash; "\n")
- `output_sep` &mdash; Separator between ICL inputs and ICL outputs (default &mdash; ". ")
- `feature_sep` &mdash; Separator between blocks of <feature_name, feature_value> pairs (default &mdash; ", ")
- `value_sep` &mdash; Separator between feature name and feature value (default &mdash; "is")
- `test_sep` &mdash; Separator between ICL samples and test input (default &mdash; "\nThe last sample &mdash;\n")
- `prompt_ID` &mdash; The ID of the prompt (default &mdash; "simple_0")
- `n_round` &mdash; Number of decimal places to round floats to in the prompt (default &mdash; 5)

#### Lime Parameters

The `lime_params` dictionary contains the following parameters:

- `kernel_width` &mdash; Width of LIME kernel (default &mdash; 0.75).
- `variance` &mdash; Variance of perturbations used (default &mdash; 0.1).
- `mode` &mdash; Data mode (default &mdash; "tabular").
- `sample_around_instance` &mdash; Boolean to determine how samples are produced (default &mdash; true).
- `n_samples` &mdash; Number of perturbations to use in LIME (default &mdash; 4000).
- `discretize_continuous` &mdash; Boolean to discretize continuous features (default &mdash; false).
- `categorical_features` &mdash; Indices of categorical/one-hot features. Default value is specific to COMPAS (default &mdash; [3, 4, 5]).

#### Experiment Parameters

The `experiment_params` dictionary contains the following parameters:

- `use_min` &mdash; Append minute of experiment start time into the experiment ID (default &mdash; true).
- `use_sec` &mdash; Append seconds of experiment start time into the experiment ID (default &mdash; true).
- `use_microsec` &mdash; Append microseconds of experiment start time into the experiment ID (default &mdash; false).

### Faithfulness Analysis

The parameters used for evaluating faithfulness metrics are From within these file, users have the ability to control the following parameters:
- `SEED` &mdash; seed value for reproducibility (default: 0).
- `data_name` &mdash; name of dataset to use, e.g., "compas", "adult", etc. (default: "compas").
- `feature_types` &mdash; list of feature types, where "c" represents continuous and "d" represents discrete (e.g. ["c", "c", "c", "d", "d", "d"]).
- `data_scaler` &mdash; data scaler method, e.g., "minmax" (default: "minmax").
- `model_name` &mdash; name of the model to use, e.g., "lr" (default: "lr").
- `model_file_name` &mdash; file name of the saved model (e.g. "<model_name>.pt").
- `model_dir` &mdash; directory of the saved model (default: "models/LR/").
- `LLM_topks_path` &mdash; path to the LLM top-Ks file (e.g.: "./outputs/LLM_QueryAndReply/<experiment_ID>/<LLM_topk>.pkl").
- `output_dir` &mdash; directory to save LLM results to (e.g.: "./outputs/LLM_QueryAndReply/<experiment_ID>/").
- `eval_min_idx` &mdash; the minimum index for evaluation (default: 0).
- `eval_max_idx` &mdash; the maximum index for evaluation (default: 400).
- `perturbation_mean` &mdash; mean of the perturbation (default: 0.0).
- `perturbation_std` &mdash; standard deviation of the perturbation (default: 0.05).
- `perturbation_flip_percentage` &mdash; percentage of perturbed samples with flipped features (default: 0.05).
- `perturbation_max_distance` &mdash; maximum distance for perturbation (default: 0.4).
- `perturb_num_samples` &mdash; number of perturbed samples (default: 100).

