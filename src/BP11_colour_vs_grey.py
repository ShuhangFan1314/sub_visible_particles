import statistics
import pandas as pd
from BP08_result_analysis import filter_results


def colour_grey_accuracy(path, model_order):
    """
    Compare the accuracy of colour vs. greyscale images for different models and save the results in a CSV file.
    """
    all_result = filter_results(path)

    # Create dictionary that will have key: model, value: parameters and accuracy
    model_dict = {}
    for file in all_result:
        model_parameter = file.split('_')
        model_dict.setdefault((model_parameter[1], model_parameter[2]), []).append((float(model_parameter[-1][:-4]), file))
    model_dict = {key: sorted(values, key=lambda x: x[0], reverse=True) for key, values in model_dict.items()}

    # Mapping of model short names to full names
    model_names = {'resnet50': 'ResNet-50', 'resnet34': 'ResNet-34', 'resnet18': 'ResNet-18',
                   'resnet50-wide': 'WRN-50-2', 'convnext-base': 'ConvNeXt-B',
                   'convnext-tiny': 'ConvNeXt-T', 'densenet121': 'DenseNet-121', 'vitb16': 'ViT-B/16'}

    # Dictionary to create table comparing 'colour vs grey' accuracy for each model
    output_dict = {'Model': [], 'colour': [], 'grey': [], 'Difference': []}

    for model in model_order:
        output_dict['Model'].append(model_names[model])
        col_best_acc = model_dict[('colour', model)][0][0] * 100
        grey_best_acc = model_dict[('grey', model)][0][0] * 100

        output_dict['colour'].append(col_best_acc)
        output_dict['grey'].append(grey_best_acc)
        output_dict['Difference'].append(col_best_acc-grey_best_acc)

    # Calculate mean and standard deviation for each model
    output_dict['Model'].append('Mean (SD)')
    output_dict['colour'].append(f'{round(statistics.mean(output_dict["colour"]), 1)}% ({round(statistics.stdev(output_dict["colour"]), 1)}%)')
    output_dict['grey'].append(f'{round(statistics.mean(output_dict["grey"]), 1)}% ({round(statistics.stdev(output_dict["grey"]), 1)}%)')
    output_dict['Difference'].append(f'{round(statistics.mean(output_dict["Difference"]), 1)}% ({round(statistics.stdev(output_dict["Difference"]), 1)}%)')

    # Create a DataFrame and save it to a CSV file
    df = pd.DataFrame(output_dict)
    df.to_csv(path + '/' + 'colour_grey.csv', index=False)


if __name__ == '__main__':
    train_out_path = '../train_out/'

    model_order = ['resnet18', 'resnet34', 'resnet50', 'resnet50-wide', 'densenet121', 'convnext-base', 'convnext-tiny', 'vitb16']

    colour_grey_accuracy(train_out_path, model_order)


