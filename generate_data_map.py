import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import argparse
import hashlib
import base64
import matplotlib.gridspec as gridspec

def load_training_dynamics(checkpoint_dir):
    """
    Load and combine training dynamics from all `training_dynamics.json` files.

    Args:
        checkpoint_dir (str): Path to checkpoint directory.

    Returns:
        dict: Combined training dynamics.
    """
    combined_dynamics = defaultdict(lambda: {"confidence": [], "probabilities": [], "correctness": [], "epoch": []})

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
                        combined_dynamics[example_id]["epoch"].extend(
                            dynamics["epoch"].get(example_id, [])
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
    processed_data = {"confidence": [], "variability": [], "correctness": [], "hashes": []}

    for example_id, metrics in combined_dynamics.items():
        avg_confidence = np.mean(metrics["confidence"]) if metrics["confidence"] else 0
        avg_variability = np.std(metrics["confidence"]) if metrics["confidence"] else 0
        avg_correctness = np.mean(metrics["correctness"]) if metrics["correctness"] else 0

        # Use example_id directly as the hash
        processed_data["confidence"].append(avg_confidence)
        processed_data["variability"].append(avg_variability)
        processed_data["correctness"].append(avg_correctness)
        processed_data["hashes"].append(example_id)

    return processed_data


def plot_data_map(fig,ax,data, title="Data Map"):
    """
    Plot the data map with confidence, variability, and correctness using hoverable hashes.

    Args:
        data (dict): Data containing confidence, variability, correctness, and hashes.
        title (str): Title for the plot.
    """
    confidence = np.array(data["confidence"])
    variability = np.array(data["variability"])
    correctness = np.array(data["correctness"])
    hashes = np.array(data["hashes"])

   

    #fig, ax = plt.subplots(figsize=(12, 8))

    # Define the ranges and corresponding markers
    ranges = [
        (0, 0, "x","red"),         # Exact 0
        (0, 0.2, "+","orange"),       # [0, 0.2)
        (0.2, 0.3, "*", "purple"),     # [0.2, 0.3)
        (0.3, 0.5, "s", "blue"),     # [0.3, 0.5)
        (0.5, 1, "o", "green"),       # [0.5, 1]
    ]

    scatter_objects = []
    used_labels = set() 

    for i in range(len(correctness)):
        # Get values for the current point
        x = variability[i]
        y = confidence[i]
        c = correctness[i]
        
        label = 0  # Default label is None
        marker = "x"  # Default marker

        # Determine the range and marker for the current correctness value
        for lower, upper, marker, color in ranges:
             if lower == upper and c == lower:  # Handle exact match for correctness == 0
                label = upper
                break
             elif lower < c <= upper:  # Handle ranges
                label = upper
                break
        
        # Only add the label if it hasn't been used before
        if label in used_labels:
            label = None
        else:
            used_labels.add(label)
        
        # Plot the point with the determined marker and label
        sc = ax.scatter(
            x,
            y,
            label=label,  # Use the predefined label
            marker=marker,
            color=color,
            s=50,
            alpha=0.8,
            edgecolors="k" if marker not in ["+", "x"] else None,
            picker=True  # Enable picking for hover
        )
        scatter_objects.append(sc)

    # Create a legend with unique labels
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

    
    # Create hover annotation
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)

    def update_annot(ind, scatter_obj):
        idx = ind["ind"][0]
        pos = scatter_obj.get_offsets()[idx]
        annot.xy = pos
        hash_idx = base64.b64decode(hashes[idx]).decode('utf-8').split("|||")
        
        annot.set_text(f"Premise: {hash_idx[0]} \nHypothesis: {hash_idx[1]} \nCorrectness: {correctness[idx]}")
        annot.get_bbox_patch().set_alpha(0.8)

    def hover(event):
        vis = annot.get_visible()
        for scatter_obj in scatter_objects:
            cont, ind = scatter_obj.contains(event)
            if cont:
                update_annot(ind, scatter_obj)
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    ax.set_xlabel("Variability")
    ax.set_ylabel("Confidence")
    ax.set_title(title)
    ax.legend(title="Correctness")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    #plt.show()


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

    
