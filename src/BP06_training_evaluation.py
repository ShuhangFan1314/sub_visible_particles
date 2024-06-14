import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def train_one_epoch(model, train_loader, optimizer, criterion, device_id, scheduler):
    """
    Train the model for one epoch.
    """
    # Put model into train mode
    model = model.train()
    for train_index, (train_data, train_labels, stress, drug) in enumerate(train_loader):
        # --- Training begins --- #
        # Send data to gpu, if there is GPU
        if torch.cuda.is_available():
            train_data = train_data.cuda(device_id)
            train_labels = train_labels.cuda(device_id)

        # Zero gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(train_data)
        # Calculate loss
        loss = criterion(outputs, train_labels)
        # Backward pass
        loss.backward()
        # Update gradients
        optimizer.step()
        # --- Training ends --- #

    # Step the scheduler
    scheduler.step()


def make_prediction(model, data_loader, device_id):
    """
    Make predictions using the trained model.
    """
    prediction_list = []
    label_list = []
    stress_list = []
    drug_list = []

    # Put model into evaluation mode
    model = model.eval()

    for data_index, (data, labels, stress, drug) in enumerate(data_loader):
        # Send data to gpu
        if torch.cuda.is_available():
            data = data.cuda(device_id)

        # Forward pass without gradients
        with torch.no_grad():
            outputs = model(data)

        # If the model is in GPU, get data to cpu
        if torch.cuda.is_available():
            outputs = outputs.cpu()

        # Add predictions and labels to respective lists
        preds = torch.argmax(outputs, dim=1)
        label_list.extend(labels.tolist())
        prediction_list.extend(preds.tolist())
        stress_list.extend(stress)
        drug_list.extend(drug)

    return np.array(prediction_list), np.array(label_list), np.array(stress_list), np.array(drug_list)


def confusion_metrics(labels, preds, average=None):
    """
    Calculate confusion matrix metrics for model evaluation
    """
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average=average)
    recall = recall_score(labels, preds, average=average)
    f1 = f1_score(labels, preds, average=average)
    return [accuracy, precision, recall, f1]
