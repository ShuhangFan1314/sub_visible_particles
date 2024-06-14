import os
import glob
import torch
import random
import pandas as pd
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch import optim
from datetime import datetime
from torch.utils.data import DataLoader
from BP05_cls_dataset import SubParDataset
from BP06_training_evaluation import confusion_metrics
from BP06_training_evaluation import train_one_epoch, make_prediction


def get_model(name, pretrained=True):
    """
    Load and prepare a model from torchvision with specified name and pretraining option.
    """
    # Mapping model names to their respective functions in torchvision
    model_dict = {
        'resnet50': models.resnet50,
        'resnet34': models.resnet34,
        'resnet18': models.resnet18,
        'resnet50-wide': models.wide_resnet50_2,
        'convnext-base': models.convnext_base,
        'convnext-tiny': models.convnext_tiny,
        'densenet121': models.densenet121,
        'vitb16': models.vit_b_16
    }

    if name not in model_dict:
        raise ValueError('Unknown model')

    # Load the model
    model = model_dict[name](pretrained=pretrained)

    # Modify the final layer to match the number of output classes (2)
    if 'resnet' in name:
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif 'convnext' in name:
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2) 
    elif 'densenet' in name:
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    elif 'vitb' in name:
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)

    return model


def calculate_drug_accuracy(prediction_test, label_test, stress_test, drug_test,
                            prediction_train, label_train, stress_train, drug_train):
    """
    Calculate and return a DataFrame containing the accuracy of predictions for different stress and drug types.
    """
    # Combine stress and drug test data, remove duplicates, and sort
    unique_stress_drug = sorted(set(zip(stress_test, drug_test)))

    # Will be used to create output DataFrame
    # Keys indicate row, values indicates list containing numbers for each column
    output_dict = {item: [0, 0, 0, 0, 0, 0] for item in unique_stress_drug}

    # Iterate through test data
    for pred, label, stress, drug in zip(prediction_test, label_test, stress_test, drug_test):
        output_dict[(stress, drug)][0] += 1
        if pred == label:
            output_dict[(stress, drug)][1] += 1

    # Iterate through train data
    for pred, label, stress, drug in zip(prediction_train, label_train, stress_train, drug_train):
        output_dict[(stress, drug)][3] += 1
        if pred == label:
            output_dict[(stress, drug)][4] += 1

    # Calculate test and train accuracy
    for key in output_dict:
        test_total, test_correct = output_dict[key][:2]
        train_total, train_correct = output_dict[key][3:-1]

        test_accuracy = test_correct / test_total if test_total != 0 else 0
        train_accuracy = train_correct / train_total if train_total != 0 else 0

        output_dict[key][2] = test_accuracy
        output_dict[key][5] = train_accuracy

    # Create DataFrame with index, column MultiIndex
    index = pd.MultiIndex.from_tuples(output_dict.keys())
    columns = pd.MultiIndex.from_product([['test', 'train'], ['total', 'correct', 'TPR']])
    df = pd.DataFrame(output_dict.values(), index=index, columns=columns)

    # Sort DataFrame based on stress type and TPR
    sorted_df = pd.DataFrame()
    unique_stress = sorted(set(stress_test))
    for stress_type in unique_stress:
        stress_subset = df[df.index.get_level_values(0) == stress_type]
        sorted_subset = stress_subset.sort_values(by=('test', 'TPR'), ascending=False)
        sorted_df = pd.concat([sorted_df, sorted_subset])

    return sorted_df


def run_train(data_path, particle_L, image_type, model_name, pretrained,
              learning_rate, momentum, batch_size, num_epoch, device_id, seed):
    """
    Train and evaluate a model with given parameters.
    """
    # Initialization
    random.seed(seed)

    # Particles used in model, will be used to name the trained model
    particles = ''
    for i in particle_L:
        particles += i + 'v'
    particles = particles[:-1]

    if pretrained:
        pretrain_label = 'pretrained'
    else:
        pretrain_label = 'untrained'

    # Check if image is grey or colour
    is_grey = False
    if image_type == 'grey':
        is_grey = True
    if image_type not in ['grey', 'colour']:
        raise ValueError('image_type: wrong label')

    initial_learning_rate = learning_rate
    
    initial_momentum = momentum

    # Time stamp
    now = datetime.now()
    start_time = now.strftime('%Y-%m-%d-%H-%m')

    # Temporary PATH to save model
    PATH = f'../train_out/{particles}_{image_type}_{model_name}_{pretrain_label}_{learning_rate}_{momentum}_{batch_size}_{num_epoch}_{start_time}.pth'

    # Augmentation
    if pretrained:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])

    aug = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    # Define datasets and data loaders
    train_dataset_path_label = []
    test_dataset_path_label = []

    # Loop to create (particle_type, label) for all particles used.
    for j in range(len(particle_L)):
        particle_train_path = glob.glob(data_path + '/train_ds/' + f'/{particle_L[j]}*')
        particle_test_path = glob.glob(data_path + '/test_ds/' + f'/{particle_L[j]}*')

        train_dataset_path_label.append((particle_train_path[0], j))
        test_dataset_path_label.append((particle_test_path[0], j))

    tr_dataset = SubParDataset(train_dataset_path_label, aug, is_grey)
    tr_loader = DataLoader(dataset=tr_dataset, batch_size=batch_size,
                           shuffle=True, num_workers=8)

    ts_dataset = SubParDataset(test_dataset_path_label, aug, is_grey)
    ts_loader = DataLoader(dataset=ts_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=8)

    # Model
    model = get_model(model_name, pretrained)
    
    if torch.cuda.is_available():
        model = model.cuda(device_id)

    # Loss
    # For additional losses see: https://pytorch.org/docs/stable/nn.html
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    # For additional optimizers see: https://pytorch.org/docs/stable/optim.html
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=0)

    # Initialize header of the output
    output_columns = ['Epoch', 'Accuracy_Test', 'Precision_Test', 'Recall_Test', 'F1 Score_Test', 'LR_Test',
                      'Accuracy_Train', 'Precision_Train', 'Recall_Train', 'F1 Score_Train', 'LR_Train']

    # Will be used to store results at each epoch
    output_data = []

    # Will be used for early break
    best_accuracy = 0.0
    no_improvement_count = 0
    best_epoch = 0
    output_df2 = pd.DataFrame()

    print('Current parameters:', particle_L, image_type, model_name, pretrained, learning_rate, momentum, batch_size, num_epoch)

    # Start training model
    for epoch in range(num_epoch):
        print('Epoch: ', epoch, 'starts')

        # Update Learning Rate
        learning_rate = scheduler.get_last_lr()[0]

        # Train the model
        train_one_epoch(model, tr_loader, optimizer, criterion, device_id, scheduler)

        # Make prediction and Calculate Metrics
        ts_preds, ts_labels, ts_stress_types, ts_drug_types = make_prediction(model, ts_loader, device_id)
        test_score = confusion_metrics(ts_preds, ts_labels, average='micro')

        tr_preds, tr_labels, tr_stress_types, tr_drug_types = make_prediction(model, tr_loader, device_id)
        train_score = confusion_metrics(tr_preds, tr_labels, average='micro')

        # Check if validation accuracy improved
        if test_score[0] > best_accuracy:
            best_accuracy = test_score[0]
            best_epoch = epoch
            no_improvement_count = 0

            # Save the model if it's the best so far
            torch.save(model.state_dict(), PATH)
            output_df2 = calculate_drug_accuracy(ts_preds, ts_labels, ts_stress_types, ts_drug_types,
                                                 tr_preds, tr_labels, tr_stress_types, tr_drug_types)
        else:
            no_improvement_count += 1

        output_data.append([epoch, test_score[0], test_score[1], train_score[2], train_score[3], learning_rate,
                            train_score[0], train_score[1], train_score[2], train_score[3], learning_rate])

        print('Epoch: ', epoch, 'ends')

        # Check for early stopping
        if no_improvement_count >= 10:
            print(f"No improvement in accuracy for 10 epochs. Stopping training.")
            break

    output_file_name = f"../train_out/{particles}_{image_type}_{model_name}_{pretrain_label}_{initial_learning_rate}_{initial_momentum}_{batch_size}_{best_epoch}_{start_time}_"

    # Convert the output data to a DataFrame, csv files
    output_df1 = pd.DataFrame(output_data, columns=output_columns)
    output_df1.to_csv(output_file_name + str(round(best_accuracy, 3)) + '.csv', index=False)
    output_df2.to_csv(output_file_name + 'drugs.csv', index=True)
    PATH_new_name = f'../train_out/{particles}_{image_type}_{model_name}_{pretrain_label}_{initial_learning_rate}_{initial_momentum}_{batch_size}_{best_epoch}_{start_time}.pth'
    os.rename(PATH, PATH_new_name)


if __name__ == '__main__':
    data_path = '../data/processed_images_V5'
    particle_L = ['heat', 'mech']
    image_type_option = ['grey', 'colour']
    model_name_options = ['resnet18', 'resnet34', 'resnet50', 'resnet50-wide', 'densenet121',
                          'vitb16', 'convnext-base', 'convnext-tiny']
    pretrained_options = [True, False]
    learning_rate_options = [0.0001, 0.0005, 0.001]
    momentum_options = [0.9, 0.95, 0.99]
    batch_size_options = [32, 64]
    num_epoch = 100
    device_id = 2
    seed = 42

    for batch_size in batch_size_options:
        for learning_rate in learning_rate_options:
            for momentum in momentum_options:
                for model_name in model_name_options:
                    for pretrained in pretrained_options:
                        for image_type in image_type_option:
                            run_train(data_path, particle_L, image_type, model_name, pretrained, learning_rate, momentum, batch_size, num_epoch, device_id, seed)
