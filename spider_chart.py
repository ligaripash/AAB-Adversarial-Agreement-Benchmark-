import numpy as np
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('TkAgg')
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


# ==========================================
# 1. DATA CONFIGURATION
# ==========================================
# The 5 Personas (Axes of the Chart)
#labels = ['Doubter', 'Sophistry', 'Authority', 'Consensus', 'Bureaucrat']

labels = ['Victim', 'Doubter', 'Sophistry', 'Authority', 'Consensus', 'Bureaucrat']

# Your ESS Scores (0-100)
# Replace these values with your actual JSON analysis results
models_data = {
    'Mistral': [74, 61, 46, 35, 16, 14],  # Strong Logic, Weak Social
    'LLAMA3': [82, 59, 54, 36, 24, 15],
    'DEEPSEEK-R1': [92, 88, 85, 76, 72, 46],
    'QWEN_2.5': [97, 95, 89, 73, 66, 47]
}

# Colors for the models
#colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

# Expanded palette (Blue, Orange, Green, Red, Purple, Brown, Pink, Gray, Olive, Cyan)
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]
# ==========================================
# 2. PLOTTING FUNCTION
# ==========================================
def make_spider_chart():
    # Number of variables
    num_vars = len(labels)

    # Compute angle of each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "close the loop"
    # We append the start point to the end of the list
    angles += angles[:1]

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    # Helper to close the data loop
    def close_loop(values):
        return values + values[:1]

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], labels, color='black', size=12, weight='bold')

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=10)
    plt.ylim(0, 100)

    # Plot each model
    for idx, (name, values) in enumerate(models_data.items()):
        data = close_loop(values)

        # Plot the line
        ax.plot(angles, data, color=colors[idx], linewidth=2, label=name)

        # Fill the area
        ax.fill(angles, data, color=colors[idx], alpha=0.15)

    # ==========================================
    # 3. STYLING
    # ==========================================
    #plt.title('Epistemic Stability (ESS) by Persona', size=20, weight='bold', y=1.1)

    # Legend placement
    #plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=12)
    plt.legend(loc='lower center', bbox_to_anchor=(1, 0.9), fontsize=12)

    # Add a subtle grid
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.show()


if __name__ == "__main__":
    make_spider_chart()