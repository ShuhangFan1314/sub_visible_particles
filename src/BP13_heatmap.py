import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def extract_value_from_latex(s):
    """
    Extract a floating-point value from a LaTeX-formatted string.
    """
    if isinstance(s, str):
        s = s.replace('\\', '').replace('$', '').replace('%', '').strip()
        accuracy = float(s)
        return accuracy


# Extract DenseNet-121 data from CSV files
def extract_densenet121_data(heat_source, mech_source):
    """
    Extract accuracy data for DenseNet-121 from CSV files and format it for heatmap plotting.
    """
    heat_df = pd.read_csv(heat_source)
    mech_df = pd.read_csv(mech_source)

    # Extract DenseNet-121 data
    heat_densenet121_colour = heat_df[heat_df['model'] == 'DenseNet-121'].iloc[0]
    heat_densenet121_grey = heat_df[heat_df['model'] == 'DenseNet-121'].iloc[1]
    mech_densenet121_colour = mech_df[mech_df['model'] == 'DenseNet-121'].iloc[0]
    mech_densenet121_grey = mech_df[mech_df['model'] == 'DenseNet-121'].iloc[1]

    colour = {}
    grey = {}

    # Extract accuracy values for images
    for image_type, heat_data, mech_data in [('colour', heat_densenet121_colour, mech_densenet121_colour),
                                             ('grey', heat_densenet121_grey, mech_densenet121_grey)]:
        for data, label in [(heat_data, ' (Heat)'), (mech_data, ' (Mech)')]:
            for protein in data.index:
                if protein not in ['Type', 'model']:
                    eval(image_type)[protein.title() + label] = extract_value_from_latex(data[protein])

    return colour, grey


def generate_heatmap(colour, grey, output_path):
    """
    Generate and display a heatmap based on the provided colour and grey data.
    """
    # Create DataFrame
    protein_order = [
        'Nivolumab (Heat)', 'Rituximab (Heat)', 'Daratumumab (Heat)', 'Infliximab (Heat)', 'Trastuzumab (Heat)',
        'Nivolumab (Mech)', 'Rituximab (Mech)', 'Bevacizumab (Mech)', 'Pertuzumab (Mech)', 'Pembrolizumab (Mech)'
    ]

    data = {
        'Protein Type': protein_order,
        'Greyscale': [grey.get(protein, np.nan) for protein in protein_order],
        'Colour': [colour.get(protein, np.nan) for protein in protein_order]
    }

    df = pd.DataFrame(data)
    df = df.set_index('Protein Type')

    # Reshape DataFrame for heatmap
    heatmap_data = df.T

    # Create annotations
    annotations = heatmap_data.applymap(lambda x: f'{x:.1f}%' if not pd.isna(x) else 'N/A')

    # Create Heatmap
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(heatmap_data, annot=annotations, fmt="", cmap='viridis', cbar_kws={'label': 'Accuracy (%)'})
    plt.title('Accuracy of DenseNet-121 for Different Protein Types', fontweight='bold', fontsize=14, pad=20)
    ax.set_xticklabels(protein_order)
    plt.xticks(rotation=50, ha='right')
    plt.xlabel('Protein Type', fontweight='bold', fontsize=12, labelpad=10)
    plt.ylabel('Image Type', fontweight='bold', fontsize=12, labelpad=10)
    plt.subplots_adjust(top=0.8, bottom=0.5, left=0.1, right=1.0, hspace=0.2, wspace=0.2)

    # Save and show the heatmap
    plt.savefig(os.path.join(output_path, 'heatmap_protein_types.svg'), dpi=500)
    plt.show()


# Extract data
colour, grey = extract_densenet121_data('../train_out/heat_table.csv', '../train_out/mech_table.csv')

# Generate heatmap
figures_path = '../figures'
generate_heatmap(colour, grey, figures_path)
