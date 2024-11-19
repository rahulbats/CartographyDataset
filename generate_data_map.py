import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import argparse


def load_training_dynamics(checkpoint_dir):
    """
    Load and combine training dynamics from all `training_dynamics.json` files.

    Args:
        checkpoint_dir (str): Path to checkpoint directory.

    Returns:
        dict: Combined training dynamics.
    """
    combined_dynamics = defaultdict(lambda: {"confidence": [], "variability": [], "correctness": []})

    for root, _, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith("training_dynamics.json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    dynamics = json.load(f)

                    # Process example IDs
                    for example_id in dynamics["example_ids"]:
                        # Populate confidence
                        if example_id in dynamics["confidence"]:
                            combined_dynamics[example_id]["confidence"].extend(
                                dynamics["confidence"].get(example_id, [])
                            )
                        # Populate variability
                        if example_id in dynamics["variability"]:
                            combined_dynamics[example_id]["variability"].extend(
                                dynamics["variability"].get(example_id, [])
                            )
                        # Populate correctness
                        if example_id in dynamics["correctness"]:
                            combined_dynamics[example_id]["correctness"].extend(
                                dynamics["correctness"].get(example_id, [])
                            )
    return combined_dynamics


def compute_metrics(combined_dynamics):
    """
    Compute average metrics for each example across epochs.

    Args:
        combined_dynamics (dict): Combined training dynamics.

    Returns:
        dict: Processed data for plotting (confidence, variability, correctness).
    """
    processed_data = {"confidence": [], "variability": [], "correctness": []}

    for example_id, metrics in combined_dynamics.items():
        avg_confidence = np.mean(metrics["confidence"]) if metrics["confidence"] else 0
        avg_variability = (
            np.mean([np.std(probabilities) for probabilities in metrics["variability"]])
            if metrics["variability"]
            else 0
        )
        avg_correctness = np.mean(metrics["correctness"]) if metrics["correctness"] else 0

        processed_data["confidence"].append(avg_confidence)
        processed_data["variability"].append(avg_variability)
        processed_data["correctness"].append(avg_correctness)

    return processed_data


def plot_data_map(data, title="Data Map"):
    """
    Plot the data map with confidence, variability, and correctness.

    Args:
        data (dict): Data containing confidence, variability, and correctness.
        title (str): Title for the plot.
    """
    confidence = data["confidence"]
    variability = data["variability"]
    correctness = data["correctness"]

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        variability, confidence, c=correctness, cmap="coolwarm", s=50, alpha=0.8, edgecolors="k"
    )
    plt.colorbar(scatter, label="Correctness")
    plt.title(title)
    plt.xlabel("Variability")
    plt.ylabel("Confidence")
    plt.grid(alpha=0.3)

    # Optional: Annotate plot regions
    plt.text(0.1, 0.9, "Easy-to-learn", transform=plt.gca().transAxes, fontsize=12, color="green")
    plt.text(0.7, 0.2, "Hard-to-learn", transform=plt.gca().transAxes, fontsize=12, color="red")
    plt.text(0.4, 0.5, "Ambiguous", transform=plt.gca().transAxes, fontsize=12, color="blue")

    plt.tight_layout()
    plt.show()
    plt.gcf().canvas.flush_events()


def main(checkpoint_dir):
    """
    Main function to load training dynamics, compute metrics, and plot the data map.

    Args:
        checkpoint_dir (str): Path to the checkpoint directory.
    """
    dynamics = load_training_dynamics(checkpoint_dir)
    processed_data = compute_metrics(dynamics)
    plot_data_map(processed_data, title="Training Data Map")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a data map from training dynamics.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory containing training_dynamics.json files.",
    )
    args = parser.parse_args()

    main(args.checkpoint_dir)
