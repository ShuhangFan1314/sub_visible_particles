import os
import statistics
import pandas as pd
from BP08_result_analysis import analyse_best_parameters


def find_drug_file(accuracy_file):
    """
    Generate the corresponding drug file name from the accuracy file name.
    """
    conditions = accuracy_file.split('_')
    drug_file = conditions[:-1] + ['drugs.csv']
    drug_file = '_'.join(drug_file)

    return drug_file


def create_tpr_table(train_out_path, model_order, drug_order_heat, drug_order_mech):
    """
    Create TPR (True Positive Rate) tables for heat and mech conditions and save them as CSV files.
    """
    model_names = {'resnet50': 'ResNet-50', 'resnet34': 'ResNet-34', 'resnet18': 'ResNet-18',
                   'resnet50-wide': 'WRN-50-2', 'convnext-base': 'ConvNeXt-B',
                   'convnext-tiny': 'ConvNeXt-T', 'densenet121': 'DenseNet-121', 'vitb16': 'ViT-B/16'}

    # Create tables for heat and mech each
    output_heat_dict = {'Type': [], 'model': []}
    output_mech_dict = {'Type': [], 'model': []}

    # Fetch best performing parameters for each model
    best_parameter_files = analyse_best_parameters(train_out_path, model_order)

    # Process colour images
    output_heat_dict['Type'].append('Colour')
    output_mech_dict['Type'].append('Colour')
    tprs_heat = {}
    tprs_mech = {}
    for index, row in best_parameter_files.iterrows():
        model = row['model']
        output_heat_dict['model'].append(model_names[model])
        output_mech_dict['model'].append(model_names[model])
        output_heat_dict['Type'].append(None)
        output_mech_dict['Type'].append(None)

        acc_file_col = row['pretrained_colour']
        drug_file_col = find_drug_file(acc_file_col)

        drug_col_df = pd.read_csv(train_out_path + '/' + drug_file_col)
        extracted_col_df = drug_col_df.iloc[1:, [0, 1, 4]]

        for index2, row2 in extracted_col_df.iterrows():
            if row2[0] == 'heat':
                output_heat_dict.setdefault(row2[1], []).append('{}%'.format(round(float(row2[2])*100, 1)))
                tprs_heat.setdefault(row2[1], []).append(float(row2[2]))
            elif row2[0] == 'mech':
                output_mech_dict.setdefault(row2[1], []).append('{}%'.format(round(float(row2[2])*100, 1)))
                tprs_mech.setdefault(row2[1], []).append(float(row2[2]))
            else:
                pass

    # Calculate mean and standard deviation for colour images
    output_heat_dict['model'].append('Mean (SD)')
    output_mech_dict['model'].append('Mean (SD)')
    for key, value in tprs_heat.items():
        mean = round(statistics.mean(value)*100, 1)
        sd = round(statistics.stdev(value)*100, 1)
        output_heat_dict[key].append(f'{mean}% ({sd}%)')

    for key, value in tprs_mech.items():
        mean = round(statistics.mean(value)*100, 1)
        sd = round(statistics.stdev(value)*100, 1)
        output_mech_dict[key].append(f'{mean}% ({sd}%)')

    # Process greyscale images
    output_heat_dict['Type'].append('Grey')
    output_mech_dict['Type'].append('Grey')
    tprs_heat = {}
    tprs_mech = {}

    for index3, row3 in best_parameter_files.iterrows():
        model = row3['model']
        output_heat_dict['model'].append(model_names[model])
        output_mech_dict['model'].append(model_names[model])
        output_heat_dict['Type'].append(None)
        output_mech_dict['Type'].append(None)

        acc_file_grey = row3['pretrained_grey']
        drug_file_grey = find_drug_file(acc_file_grey)

        drug_grey_df = pd.read_csv(train_out_path + '/' + drug_file_grey)
        extracted_grey_df = drug_grey_df.iloc[1:, [0, 1, 4]]

        for index4, row4 in extracted_grey_df.iterrows():
            if row4[0] == 'heat':
                output_heat_dict.setdefault(row4[1], []).append('{}%'.format(round(float(row4[2])*100, 1)))
                tprs_heat.setdefault(row4[1], []).append(float(row4[2]))
            elif row4[0] == 'mech':
                output_mech_dict.setdefault(row4[1], []).append('{}%'.format(round(float(row4[2])*100, 1)))
                tprs_mech.setdefault(row4[1], []).append(float(row4[2]))
            else:
                pass

    # Calculate mean and standard deviation for greyscale images
    output_heat_dict['model'].append('Mean (SD)')
    output_mech_dict['model'].append('Mean (SD)')

    for key, value in tprs_heat.items():
        mean = round(statistics.mean(value) * 100, 1)
        sd = round(statistics.stdev(value) * 100, 1)
        output_heat_dict[key].append(f'{mean}% ({sd}%)')

    for key, value in tprs_mech.items():
        mean = round(statistics.mean(value) * 100, 1)
        sd = round(statistics.stdev(value) * 100, 1)
        output_mech_dict[key].append(f'{mean}% ({sd}%)')

    # Create DataFrames and save as CSV files
    output_heat_dict = {key: output_heat_dict[key] for key in drug_order_heat}
    df_heat = pd.DataFrame(output_heat_dict)
    output_mech_dict = {key: output_mech_dict[key] for key in drug_order_mech}
    df_mech = pd.DataFrame(output_mech_dict)

    df_heat.to_csv(os.path.join(train_out_path, 'heat_table.csv'), index=False)
    df_mech.to_csv(os.path.join(train_out_path, 'mech_table.csv'), index=False)


if __name__ == '__main__':
    train_out_path = '../train_out/'

    model_order = ['resnet18', 'resnet34', 'resnet50', 'resnet50-wide', 'densenet121', 'convnext-base', 'convnext-tiny', 'vitb16']
    drug_order_heat = ['Type', 'model', 'nivolumab', 'rituximab', 'daratumumab', 'infliximab', 'trastuzumab']
    drug_order_mech = ['Type', 'model', 'nivolumab', 'rituximab', 'bevacizumab', 'pertuzumab', 'pembrolizumab']

    create_tpr_table(train_out_path, model_order, drug_order_heat, drug_order_mech)
