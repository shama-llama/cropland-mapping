# Cropland Mapping

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://docs.python.org/3.12/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-EE4C2C?logo=pytorch&logoColor=white)](https://docs.pytorch.org/docs/2.7/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/whats_new/v1.7.html)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shama-llama/cropland-mapping/blob/main/notebooks/cmap_msi.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project uses a deep learning framework for crop classification using multichannel inputs from Sentinel-2, combining high-resolution RGB imagery with vegetation indices (VIs) such as NDVI and GNDVI to capture both spatial and phenological crop characteristics. The model used is a convolutional long short-term memory (ConvLSTM) architecture to process time-series sequences of fused sensor data that integrates spatial features and temporal dependencies across multiple seasons. Input normalization, channel-wise sensor fusion, and data augmentation are used to mitigate class imbalance inherent in the dataset. It is evaluated on the [Canadian Cropland Dataset](https://github.com/bioinfoUQAM/Canadian-cropland-dataset-github) and the model achieves ~90% classification accuracy.

## Architecture

![Model Architecture](/diagram/diagram.png)

## Findings

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| BARLEY | 0.82 | 0.75 | 0.78 | 36 |
| CANOLA | 0.94 | 0.97 | 0.96 | 66 |
| CORN | 0.89 | 0.91 | 0.90 | 55 |
| MIXEDWOOD | 0.94 | 1.00 | 0.97 | 32 |
| OAT | 0.91 | 0.88 | 0.89 | 67 |
| ORCHARD | 0.98 | 0.92 | 0.95 | 51 |
| PASTURE | 0.86 | 0.91 | 0.89 | 47 |
| POTATO | 0.90 | 0.94 | 0.92 | 67 |
| SOYBEAN | 0.94 | 0.89 | 0.92 | 104 |
| SPRING_WHEAT | 0.56 | 0.71 | 0.62 | 7 |
| **Accuracy** | | | **0.91** | **532** |
| **Macro Avg** | **0.87** | **0.89** | **0.88** | **532** |
| **Weighted Avg** | **0.91** | **0.91** | **0.91** | **532** |

> There is visible overfitting starting from the 4th epoch onwards which should be addressed in later works.

## Project Setup

This project uses `uv` for package management. `uv` is an extremely fast Python package and project manager, written in Rust that can be used as a drop-in replacement for `pip`, `pip-tools`, `pipx`, `poetry`, `pyenv`, `twine`, `virtualenv`.

- **`uv` Installation**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

- **Clone the Repository:**

    ```bash
    git clone https://github.com/shama-llama/cropland-mapping.git
    cd cropland-mapping
    ```

- **Create a Virtual Environment and Install Dependencies with `uv`:**

    ```bash
    uv venv
    uv pip install -e .
    ```

- **Activate the Virtual Environment:**

    ```bash
    source .venv/bin/activate
    ```

- **Launch Jupyter Notebook:**

    ```bash
    jupyter notebook
    ```

    Navigate to the `notebooks/` directory to run the analysis.

## Notebooks

- `cmap_dataset.ipynb`: Data preparation with minor preprocessing and loading into HDF5 for better portability.
- `cmap_eda.ipynb`: Exploratory data analysis.
- `cmap_msi.ipynb`: ConvLSTM model training and evaluation.

## Model

The trained ConvLSTM model weights are saved in the `models/` directory.

```python
torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
```

## References

> A. A. B. Jacques, A. B. Diallo, and E. Lord, “The Canadian Cropland Dataset: A New Land Cover Dataset for Multitemporal Deep Learning Classification in Agriculture,” June 04, 2023, arXiv: arXiv:2306.00114. doi: [10.48550/arXiv.2306.00114](https://doi.org/10.48550/arXiv.2306.00114).
>
> M. O. Turkoglu et al., “Crop mapping from image time series: Deep learning with multi-scale label hierarchies,” Remote Sensing of Environment, vol. 264, p. 112603, Oct. 2021, doi: [10.1016/j.rse.2021.112603](https://doi.org/10.1016/j.rse.2021.112603).
>
> D. Darwish, “Improving Techniques for Convolutional Neural Networks Performance,” European Journal of Electrical Engineering and Computer Science, vol. 8, pp. 1–16, Jan. 2024. Number: 1.
>
> J. M. Johnson and T. M. Khoshgoftaar, “Survey on deep learning with class imbalance,” Journal of Big Data, vol. 6, pp. 1–54, Dec. 2019. Number: 1 Publisher: SpringerOpen.
>
> C. Shorten and T. M. Khoshgoftaar, “A survey on Image Data Augmentation for Deep Learning,” Journal of Big Data, vol. 6, pp. 1–48, Dec. 2019. Number: 1 Publisher: SpringerOpen.

## License

This project is licensed under the terms of the [MIT](LICENSE) open source license.
