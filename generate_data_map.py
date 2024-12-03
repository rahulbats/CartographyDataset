import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import argparse
import matplotlib.patches as mpatches
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
    combined_dynamics = defaultdict(lambda: {"confidence": [], "probabilities": [], "correctness": [], "label": [], "epoch": [], "step": []})

    for root, _, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith("training_dynamics.json"):
                file_path = os.path.join(root, file)
                with open(file_path, "r") as f:
                    dynamics = json.load(f)

                    # Process example IDs
                    for example_id in dynamics["example_ids"]:
                        
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
    print(f"Loaded {len(combined_dynamics)} training dynamics before filtering")  
    for example_id, metrics in combined_dynamics.items():
        #if len(metrics["confidence"])>1:
            avg_confidence = np.mean(metrics["confidence"]) if metrics["confidence"] else 0
            avg_variability = np.std(metrics["confidence"], dtype=np.float64) if metrics["confidence"] else 0
            avg_correctness = np.mean(metrics["correctness"]) if metrics["correctness"] else 0
            label = metrics["label"][0] if metrics["label"] else 0

            # Use example_id directly as the hash
            processed_data["confidence"].append(avg_confidence)
            processed_data["variability"].append(avg_variability)
            processed_data["correctness"].append(avg_correctness)
            processed_data["hashes"].append(example_id)
            processed_data["label"].append(label)
    print(f"Loaded {len(processed_data['confidence'])} training dynamics after filtering")  
    return processed_data


def plot_data_map(fig, ax, data, title="Data Map"):
    confidence = np.array(data["confidence"])
    variability = np.array(data["variability"])
    correctness = np.array(data["correctness"])
    labels = np.array(data["label"])
    hashes = np.array(data["hashes"])

    # Define the ranges and corresponding markers
    marker_ranges = [
       
        (0, 0.2, "red"),    # [0, 0.2)
        (0.2, 0.4,  "orange"),    # [0, 0.2)
        (0.4, 0.6,  "purple"),  # [0.2, 0.3)
        (0.6, 0.8, "green"),     # [0.5, 1]
        (0.8, 1,  "blue"),    # [0.3, 0.5)
    ]

    # Generate legend handles based on marker_ranges
    legend_handles = []
    patch = mpatches.Patch(color="red", label="0 <= Correctness ≤ 0.2")
    legend_handles.append(patch)
    
    i=1
    while i < len(marker_ranges):
        lower, upper,  color = marker_ranges[i]
        legend_label = f"{lower} < Correctness ≤ {upper}"
        patch = mpatches.Patch(color=color, label=legend_label)
        legend_handles.append(patch)
        i+=1


    #scatter_objects = []
    x_values = []
    y_values = []
    colors = []
    metadata = []
    for index in range(len(correctness)):
        if hashes[index] is None:
            continue
        color="red"
        for marker_range in marker_ranges:
            lower, upper, color = marker_range
            if(lower<=correctness[index] and correctness[index]<=upper):
                color = color
                break
            
        x_values.append(variability[index])
        y_values.append(confidence[index])
        colors.append(color)
        
        hash_idx = base64.b64decode(hashes[index]).decode('utf-8').split("|||")
        metadata.append(f"Premise: {hash_idx[0]} \nHypothesis: {hash_idx[1]} \nLabel: {'Entailment' if labels[index]==0 else 'Neutral' if labels[index]==1 else 'Contradiction' } \nCorrectness: {correctness[index]:.2f}")

   
    # Plot all points in a single scatter plot
    sc = ax.scatter(
        x_values,  # All x-values
        y_values,  # All y-values
        c=colors,  # Colors for each point
        s=50,
        alpha=0.8,
        edgecolors="k", 
    )
    sc.metadata = metadata  
    # Attach mplcursors to the single scatter plot
    cursor = mplcursors.cursor(sc, hover=True)

    @cursor.connect("add")
    def on_hover(sel):
        hovered_metadata = sc.metadata[sel.index]  # Use sel.index to get metadata
        # Change alpha for hovered point
        sel.annotation.set_bbox(dict(boxstyle="round", facecolor="yellow", edgecolor="black", alpha=1.0))  # alpha=1.0 for no transparency
        sel.annotation.set_text(
            #f"Premise: {hash_idx[0]} \nHypothesis: {hash_idx[1]} \nLabel: {labels[index]} \nCorrectness: {correctness[index]:.2f}"
            hovered_metadata
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
    ax.legend(handles=legend_handles, title="Correctness", loc="best")



    # Finalize axis limits after plotting
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Define offsets as percentages of the axis range
    x_offset = (x_max - x_min) * 0.02 # 10% of the x-range
    y_offset = (y_max - y_min) * 0.02  # 10% of the y-range

    # Add text in the top-left corner
    ax.text(x_min + x_offset, y_max - y_offset, "Easy to Learn", fontsize=10, ha='left', va='top', color='red', clip_on=False, fontweight='bold')

    # Add text in the middle
    ax.text(x_max-x_offset, (y_min + y_max) / 2, "Ambiguous", fontsize=10, ha='right', va='center', color='brown', clip_on=False, fontweight='bold')

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
    ax.hist(data, bins=30, color=color, alpha=0.8, orientation="vertical")
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


def get_focussed_sets( checkpoint_dir,min_confidence=None,max_confidence=None,min_variability=None, max_variability=None,min_correctness=None, max_correctness=None):
    """
    Filter the dataset based on confidence, variability, and correctness thresholds.

    Args:
        checkpoint_dir (str): Path to the checkpoint directory containing training_dynamics.json files.
        min_confidence (float): Minimum confidence threshold.
        max_variability (float): Maximum variability threshold.
        min_correctness (float): Minimum correctness threshold.

    Returns:
        dict: Filtered data.
    """
    dynamics = load_training_dynamics(checkpoint_dir)
    data = compute_metrics(dynamics)
    filtered_hashes ={"premise": [], "hypothesis": [], "label": []}
    print(f"Loaded {len(data['hashes'])} training dynamics before filtering")
    for idx in range(len(data["hashes"])):

        if min_variability is not None and data["variability"][idx]<= min_variability :
            continue
        if max_variability is not None and data["variability"][idx]>= max_variability :
            continue
        if min_confidence is not None and data["confidence"][idx]<= min_confidence :
            continue
        if max_confidence is not None and data["confidence"][idx]>= max_confidence :
            continue
        if min_correctness is not None and data["correctness"][idx]<= min_correctness :
            continue
        if max_correctness is not None and data["correctness"][idx]>= max_correctness :
            continue
        hash_idx = base64.b64decode(data["hashes"][idx]).decode('utf-8').split("|||")    
        filtered_hashes["premise"].append(hash_idx[0])
        filtered_hashes["hypothesis"].append(hash_idx[1])
        filtered_hashes["label"].append(data["label"][idx])
    print(f"Loaded {len(filtered_hashes['premise'])} training dynamics after filtering")    
    return filtered_hashes 
   

def plot(checkpoint_dir):
    dynamics = load_training_dynamics(checkpoint_dir)
    processed_data = compute_metrics(dynamics)
    plot_combined(processed_data, title="Training Data Map")

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
    plot(args.checkpoint_dir)
    
