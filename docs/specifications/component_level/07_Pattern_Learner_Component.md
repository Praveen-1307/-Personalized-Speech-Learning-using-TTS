# Component Specification: Pattern Learner (`personalization_engine/pattern_learner.py`)

## Overview
This component learns reusable structure from extracted voice features. It converts nested feature dictionaries into fixed-length vectors, trains either a Gaussian Mixture Model or a small autoencoder, and generates profile-oriented outputs from the learned representation.

## Input Data and Structure
| Input | Type | Structure |
| :--- | :--- | :--- |
| `features_list` | `list[dict]` | List of nested feature dictionaries from the extractor. |\n| `features_matrix` | `np.ndarray` | Matrix of flattened numeric features. |\n| `features` | `dict` | Single feature dictionary used for profile generation. |\n| `method` | `str` | `gmm` or `neural` mode configurations. |\n| `n_components` | `int` | Number of clustered GMM components. |

## Sequence of Steps
1. Flatten selected prosodic, speaking-pattern, and emotion keys into vectors.\n2. Scale the feature matrix when multiple samples are available.\n3. Train either a GMM or an autoencoder-style neural model.\n4. Compute feature-importance values when the method supports it.\n5. Generate a profile representation from a single feature dictionary.\n6. Save or reload trained state with joblib.

## Data Transformations
- `list[dict] -> np.ndarray`: nested feature dictionaries become a model matrix.\n- `np.ndarray -> scaled matrix`: normalization before learning.\n- `trained model -> profile dict`: posterior probabilities or encoded embeddings are exported.

## Output Data and Structure
| Output | Type | Structure |
| :--- | :--- | :--- |
| `feature matrix` | `np.ndarray` | One row per training sample. |\n| `learned model` | `object` | GaussianMixture or PyTorch module. |\n| `generated profile` | `dict` | Model-dependent representation plus feature vector data. |

## Notes
- Small datasets are artificially augmented with Gaussian noise before GMM fitting.\n- The learner operates on a subset of extracted features, not the entire feature tree.
