import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def box_plot_model_accuracy(image_type, file_path, output_path):
    """
    Plots a box plot of model accuracy for pretrained and non-pretrained models.
    """
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Extract the model name from the 'conditions' column
    data['model_name'] = data['conditions'].apply(lambda x: eval(x)[0])

    # Convert the data to long format for plotting (melt pretrained, untrained)
    melted_data = pd.melt(data, id_vars=['model_name'],
                          value_vars=[f'pretrained_{image_type}', f'untrained_{image_type}'],
                          var_name='Training Type', value_name='accuracy')

    # Rename training types for clarity
    melted_data['Training Type'] = melted_data['Training Type'].replace({
        f'pretrained_{image_type}': 'Pretrained',
        f'untrained_{image_type}': 'Non-Pretrained'
    })

    # Define the order of models for consistent plotting
    model_order = ['resnet18', 'resnet34', 'resnet50', 'resnet50-wide',
                   'densenet121', 'vitb16', 'convnext-base', 'convnext-tiny']

    # Define labels for models
    model_name_labels = ['ResNet-18', 'ResNet-34', 'ResNet-50', 'Wide ResNet-50-2',
                         'DenseNet-121', 'ViT-B/16', 'ConvNeXt-Base', 'ConvNeXt-Tiny']

    # Define color palette based on the image type
    if image_type == "colour":
        palette = {'Pretrained': '#6a0dad', 'Non-Pretrained': '#D2B4DE'}  # Colors for 'colour' image type
    else:
        palette = {'Pretrained': '#626567', 'Non-Pretrained': '#D0D3D4'}  # Colors for 'grey' image type

    # Create the box plot
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(x='model_name', y='accuracy', hue='Training Type', data=melted_data, order=model_order, width=0.6,
                     palette=palette)

    # Customize the plot
    plt.xlabel('Model', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold', labelpad=10)
    plt.title(f'Accuracy of {image_type.title()} Images by Model', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=50, ha='right')
    ax.set_xticklabels(model_name_labels)
    plt.ylim(0.45, 1)
    plt.subplots_adjust(top=0.918, bottom=0.4, left=0.3, right=0.7, hspace=0.0, wspace=0.0)

    # Save and show the plot
    plt.savefig(os.path.join(output_path, f'boxplot_{image_type}.svg'), dpi=500)
    plt.show()


if __name__ == '__main__':
    file_path = '../train_out/analysis_df.csv'
    figures_path = '../figures'

    # Generate the plot for colour dataset and greyscale dataset
    box_plot_model_accuracy('colour', file_path, figures_path)
    box_plot_model_accuracy('grey', file_path, figures_path)
