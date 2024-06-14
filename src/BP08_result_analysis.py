import os
import pandas as pd


def filter_results(directory):
    """
    Filter out the CSV files that represent finished training results.
    """
    all_files = os.listdir(directory)

    # Filter the finished training result files
    csv_files = [file for file in all_files
                 if (file.endswith('.csv') and len(file.split('_')) == 9
                     and file.split('_')[-1] != 'drugs.csv')
                 or (file.endswith('.csv') and len(file.split('_')) == 10
                     and file.split('_')[-1] != 'drugs.csv')]

    return csv_files


def analyse_results_df(train_out_path):
    """
    Analyze the training results and generate a summary CSV file.
    """
    output_file = os.path.join(train_out_path, f'analysis_df.csv')
    csv_files = filter_results(train_out_path)

    output_dic = {}

    # Parse data from CSV files and organize into a dictionary
    for file in csv_files:
        if len(file.split('_')) == 9:
            particles, image_type, model, pretrained, learning_rate, batch_size, epoch, date, accuracy = file.split('_')
            key = (model, float(learning_rate), int(batch_size))
        elif len(file.split('_')) == 10:
            particles, image_type, model, pretrained, learning_rate, momentum, batch_size, epoch, date, accuracy = file.split('_')
            key = (model, float(learning_rate), float(momentum), int(batch_size))
        else:
            pass

        output_dic.setdefault(key, []).append((pretrained, image_type, float(accuracy[:-4])))

    output_dic = {key: sorted(value) for key, value in output_dic.items()}

    # Convert data into a DataFrame and write to a CSV file
    df_data = {'conditions': [],
               'pretrained_colour': [], 'pretrained_grey': [],
               'untrained_colour': [], 'untrained_grey': []}

    for key, values in output_dic.items():
        df_data['conditions'].append(key)
        checker = {'pretrained_colour': True,
                   'pretrained_grey': True,
                   'untrained_colour': True,
                   'untrained_grey': True}

        for value in values:
            column_name = f'{value[0]}_{value[1]}'
            if checker[column_name]:
                df_data[column_name].append(value[2])
                checker[column_name] = False

        for key, value in checker.items():
            if value:
                df_data[key].append(None)

    df = pd.DataFrame(df_data)
    df = df.sort_values(by='pretrained_colour', ascending=False)
    df.to_csv(output_file, index=False)


def analyse_best_parameters(train_out_path, model_order):
    """
    Analyze the best parameters for each model and return a DataFrame.
    """
    csv_files = filter_results(train_out_path)

    output_dic = {}

    # Parse data from CSV files and organize into a dictionary
    for file in csv_files:
        if len(file.split('_')) == 9:
            particles, image_type, model, pretrained, learning_rate, batch_size, epoch, date, accuracy = file.split('_')
        elif len(file.split('_')) == 10:
            particles, image_type, model, pretrained, learning_rate, momentum, batch_size, epoch, date, accuracy = file.split('_')
        else:
            pass

        key = (model, image_type, pretrained)

        output_dic.setdefault(key, []).append((float(accuracy[:-4]), file))

    # Sort to find best performing parameters
    output_dic = {key: sorted(values, key=lambda x: x[0], reverse=True) for key, values in output_dic.items()}

    for key, value in output_dic.items():
        if value:  # Check if the list is not empty
            output_dic[key] = value[0]

    # Convert data into a DataFrame and write to a CSV file
    df_data = {'model': [],
               'pretrained_colour': [], 'pretrained_grey': [],
               'untrained_colour': [], 'untrained_grey': []}

    for model_name in model_order:
        df_data['model'].append(model_name)
        df_data['pretrained_colour'].append(output_dic[(model_name, 'colour', 'pretrained')][1])
        df_data['pretrained_grey'].append(output_dic[(model_name, 'grey', 'pretrained')][1])
        df_data['untrained_colour'].append(output_dic[(model_name, 'colour', 'untrained')][1])
        df_data['untrained_grey'].append(output_dic[(model_name, 'grey', 'untrained')][1])

    df = pd.DataFrame(df_data)

    return df


if __name__ == '__main__':
    path = '../train_out'

    analyse_results_df(path)
