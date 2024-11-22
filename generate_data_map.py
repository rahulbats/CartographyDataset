import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import argparse

import base64
import matplotlib.gridspec as gridspec
import mplcursors

def load_training_dynamics(checkpoint_dir):
    """
    Load and combine training dynamics from all `training_dynamics.json` files.

    Args:
        checkpoint_dir (str): Path to checkpoint directory.

    Returns:
        dict: Combined training dynamics.
    """
    combined_dynamics = defaultdict(lambda: {"confidence": [], "probabilities": [], "correctness": [], "label": []})

    for root, _, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith("training_dynamics.json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    dynamics = json.load(f)

                    # Process example IDs
                    for example_id in dynamics["example_ids"]:
                        #if( dynamics["epoch"].get(example_id)[0] in combined_dynamics[example_id]["epoch"]):
                         #   continue
                        combined_dynamics[example_id]["confidence"].extend(
                            dynamics["confidence"].get(example_id, [])
                        )
                        combined_dynamics[example_id]["probabilities"].extend(
                            dynamics["probabilities"].get(example_id, [])
                        )
                        combined_dynamics[example_id]["correctness"].extend(
                            dynamics["correctness"].get(example_id, [])
                        )
                        combined_dynamics[example_id]["label"].extend(
                            dynamics["label"].get(example_id, [])
                        )
                        
                print(f"Loaded {len(combined_dynamics)} training dynamics from {file_path}")                
    return combined_dynamics

def compute_metrics(combined_dynamics):
    """
    Compute average metrics for each example across epochs.

    Args:
        combined_dynamics (dict): Combined training dynamics.

    Returns:
        dict: Processed data for plotting (confidence, variability, correctness, hashes).
    """
    processed_data = {"confidence": [], "variability": [], "correctness": [], "hashes": [], "label": []}

    for example_id, metrics in combined_dynamics.items():
        avg_confidence = np.mean(metrics["confidence"]) if metrics["confidence"] else 0
        avg_variability = np.std(metrics["confidence"]) if metrics["confidence"] else 0
        avg_correctness = np.mean(metrics["correctness"]) if metrics["correctness"] else 0
        label = metrics["label"][0] if metrics["label"] else 0

        # Use example_id directly as the hash
        processed_data["confidence"].append(avg_confidence)
        processed_data["variability"].append(avg_variability)
        processed_data["correctness"].append(avg_correctness)
        processed_data["hashes"].append(example_id)
        processed_data["label"].append(label)

    return processed_data


def plot_data_map(fig, ax, data, title="Data Map"):
    confidence = np.array(data["confidence"])
    variability = np.array(data["variability"])
    correctness = np.array(data["correctness"])
    labels = np.array(data["label"])
    hashes = np.array(data["hashes"])

    # Define the ranges and corresponding markers
    ranges = [
        (0, 0, "x", "red"),         # Exact 0
        (0, 0.2, "+", "orange"),    # [0, 0.2)
        (0.2, 0.3, "*", "purple"),  # [0.2, 0.3)
        (0.3, 0.5, "s", "blue"),    # [0.3, 0.5)
        (0.5, 1, "o", "green"),     # [0.5, 1]
    ]

    scatter_objects = []
    for lower, upper, marker, color in ranges:
        # Filter data points based on correctness range
        indices = (correctness >= lower) & (correctness <= upper)
        sc = ax.scatter(
            variability[indices],  # x-values
            confidence[indices],   # y-values
            label=f"{lower} ≤ Correctness ≤ {upper}",
            marker=marker,
            color=color,
            s=50,
            alpha=0.8,
            edgecolors="k",
        )
        scatter_objects.append(sc)

    # Attach mplcursors to all scatter plots
    cursor = mplcursors.cursor(scatter_objects, hover=True)

    @cursor.connect("add")
    def on_hover(sel):
        index = sel.index
        hash_idx = base64.b64decode(hashes[index]).decode('utf-8').split("|||")
        sel.annotation.set_text(
            f"Premise: {hash_idx[0]} \nHypothesis: {hash_idx[1]} \nLabel: {labels[index]} \nCorrectness: {correctness[index]:.2f}"
        )

    @cursor.connect("remove")
    def on_leave(sel):
        if sel.annotation.get_figure() is not None:
           
            sel.annotation.set_visible(False)
            sel.annotation.get_figure().canvas.draw_idle()

    # Customize the plot
    ax.set_xlabel("Variability")
    ax.set_ylabel("Confidence")
    ax.set_title(title)
    ax.legend(title="Correctness")



    # Finalize axis limits after plotting
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Define offsets as percentages of the axis range
    x_offset = (x_max - x_min) * 0.10  # 10% of the x-range
    y_offset = (y_max - y_min) * 0.10  # 10% of the y-range

    # Add text in the top-left corner
    ax.text(x_min + x_offset, y_max - y_offset, "Easy to Learn", fontsize=10, ha='left', va='top', color='blue', clip_on=False, fontweight='bold')

    # Add text in the middle
    ax.text((x_min + x_max) / 2, (y_min + y_max) / 2, "Ambiguous", fontsize=10, ha='center', va='center', color='brown', clip_on=False, fontweight='bold')

    # Add text in the bottom-right corner
    ax.text(x_min + x_offset, y_min + y_offset, "Hard to Learn", fontsize=10, ha='left', va='bottom', color='black', clip_on=False, fontweight='bold')



    plt.grid(alpha=0.3)
    plt.tight_layout()



        
def plot_single_density(ax, data, label, color):
    """
    Plot a single density histogram.

    Args:
        ax (matplotlib.axis): Axis to draw the density plot.
        data (list or numpy.array): Data for the histogram.
        label (str): Label for the histogram.
        color (str): Color for the histogram bars.
    """
    ax.hist(data, bins=30, color=color, alpha=0.8, orientation="horizontal")
    ax.set_title(label)
    ax.set_xlabel("Density")
    ax.grid(alpha=0.3)


def plot_combined(data, title="Data Map with Density"):
    """
    Combine the scatter plot (data map) with three vertical density histograms on the right.

    Args:
        data (dict): Data containing confidence, variability, correctness, and hashes.
        title (str): Title for the combined plot.
    """
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])  # Scatter plot and vertical stack



    # Scatter plot on the left
    ax_scatter = plt.subplot(gs[0])
    plot_data_map(fig, ax_scatter, data)

    # Create a vertical split for density plots
    density_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1], hspace=0.4)

    # Density plot for confidence
    ax_density_conf = plt.subplot(density_gs[0])
    plot_single_density(ax_density_conf, data["confidence"], "Confidence", "purple")

    # Density plot for variability
    ax_density_var = plt.subplot(density_gs[1])
    plot_single_density(ax_density_var, data["variability"], "Variability", "teal")

    # Density plot for correctness
    ax_density_corr = plt.subplot(density_gs[2])
    plot_single_density(ax_density_corr, data["correctness"], "Correctness", "green")




    # Set overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


# Example main function and argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a data map from training dynamics.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory containing training_dynamics.json files.",
    )
    args = parser.parse_args()

    # Load dynamics, compute metrics, and plot
    dynamics = load_training_dynamics(args.checkpoint_dir)
    processed_data = compute_metrics(dynamics)
    plot_combined(processed_data, title="Training Data Map")

    
