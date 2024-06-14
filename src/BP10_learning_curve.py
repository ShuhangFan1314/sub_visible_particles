import os
import pandas as pd
import matplotlib.pyplot as plt
from BP08_result_analysis import analyse_best_parameters


def test_train_plot(train_out_path, output_path, model_order):
    """
    Generate and save plots for training and testing accuracy over epochs for different models.
    """
    # Retrieve best performing parameters for each model
    df = analyse_best_parameters(train_out_path, model_order)

    # Define conditions and their corresponding colors
    types = ['pretrained_colour', 'pretrained_grey', 'untrained_colour', 'untrained_grey']
    colour = {'pretrained_colour': 'blueviolet', 'untrained_colour': 'blueviolet',
              'pretrained_grey': 'darkslategray', 'untrained_grey': 'darkslategray'}

    for column in types:
        i, j = 0, 0
        fig, axes = plt.subplots(2, 4, figsize=(15, 7))

        for index, row in df.iterrows():
            # Read the CSV file corresponding to the current condition and model
            file_df = pd.read_csv(os.path.join(train_out_path, row[column]))

            # Plot training and testing accuracy
            axes[i][j].plot(file_df['Epoch'], file_df['Accuracy_Train'], label='Train',
                            color=colour[column], marker='o', markersize=2, linewidth=1)
            axes[i][j].plot(file_df['Epoch'], file_df['Accuracy_Test'], label='Test', linestyle='--',
                            color=colour[column], markersize=2, linewidth=1)

            axes[i][j].set_ylim(0.4, 1.0)

            # Add axis labels
            axes[i][j].set_xlabel('Epoch')
            axes[i][j].set_ylabel('Accuracy')

            # Add legend to each subplot
            axes[i][j].legend(loc='lower right')

            # Update subplot index
            j += 1
            if j == 4:
                i += 1
                j = 0

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'learning_curve_{column}.svg'), bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    train_out_path = '../train_out'
    figures_path = '../figures'
    model_order = ['resnet18', 'resnet34', 'resnet50', 'resnet50-wide', 'densenet121', 'vitb16', 'convnext-base', 'convnext-tiny']

    test_train_plot(train_out_path, figures_path, model_order)
