# Focused-Training-Cartography

This repository, **Focused-Training-Cartography**, contains code to create cartographic plots for training data with tooltips, which show the original dataset record. The primary objective is to identify and address hard-to-learn and ambiguous examples using cartographic data analysis techniques.

## Features

- **Cartographic Plots with Tooltips**: Generates cartographic maps to visualize training data points with tooltips that display the original dataset record (hypothesis, premise, and labels) for each instance.
- **Custom Trainer with Callback**: Implements a custom training loop with callbacks to capture confidence, variability, and correctness scores for each training instance.
- **Focused Training**: Runs focused training using min/max values for confidence, variability, and correctness, allowing targeted improvements.
- **Focused Dataset Creation**: Automatically creates focused datasets for fine-tuning based on criteria such as low confidence and high variability.
- **Ensemble Learning**: Combines models trained on different data subsets to improve generalization and robustness.

## Installation

To use the code in this repository, you need to have Python 3.7+ and the following dependencies:

```bash
pip install -r requirements.txt
```

The key dependencies include:

- `transformers`: HuggingFace library for training Electra-Small
- `matplotlib`: For visualizing cartographic maps
- `numpy`, `pandas`: For data handling and manipulation

## Usage

### Training the Model

To train the Electra-Small model on the SNLI dataset:

```bash
python train.py --dataset snli --model electra-small
```

### Capturing Confidence and Variability Scores

The custom trainer automatically logs confidence, variability, and correctness scores for each instance. These scores are used to create focused datasets and cartographic visualizations.

```bash
python capture_scores.py --output scores.csv
```

### Generating Cartographic Maps

To generate cartographic maps to visualize the training data points:

```bash
python visualize_cartography.py --input scores.csv --output plot_with_tooltip.png
```

This generates a plot that allows inspection of data points, including tooltips that display the hypothesis, premise, and labels for each instance.

### Fine-Tuning on Focused Dataset

To fine-tune the model on a focused subset of data using min/max confidence, variability, and correctness:

```bash
python fine_tune.py --dataset focused_subset --model electra-small --min_confidence 0.5 --max_variability 0.2
```

## Ensemble Learning

The repository includes an ensemble approach to improve robustness:

```bash
python ensemble.py --models model_1.pt model_2.pt --output ensemble_model.pt
```

## Results

The key findings from the training experiments include:

- Baseline accuracy: **89%** on SNLI dataset.
- Improved accuracy on challenging examples with focused dataset fine-tuning.
- Robust performance using an ensemble of multiple models.

## Cartographic Analysis Example



The cartographic map shows how the model interacts with different subsets of data, allowing targeted improvements.

## Citation

If you find this repository useful, please cite:

```
@article{bhattacharya2024nli,
  title={Improving Natural Language Inference with Electra-Small and Cartographic Data Analysis},
  author={Rahul Bhattacharya},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For questions or collaborations, please contact:

- Rahul Bhattacharya ([rahulbats@gmail.com](mailto\:rahulbats@gmail.com))

