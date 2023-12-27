
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

def starting_train(train_dataset, val_dataset, model: nn.Module, hyperparameters, n_eval):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = model.to(device)

    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.BCEWithLogitsLoss()

    # Compute class weights
    labels = train_dataset.get_labels()
    class_weights = torch.tensor([1.0 / (labels == 0).sum(), 1.0 / (labels == 1).sum()]).to(device)

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        model.train()

        for batch_inputs, batch_labels in train_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = loss_fn(predictions.squeeze(), batch_labels.float())

            # Apply class weights to the loss
            loss = (loss * class_weights[batch_labels.long()]).mean()

            loss.backward()
            optimizer.step()

            if step % n_eval == 0:
                print(step)
                model.eval()

                # Compute training loss and accuracy.
                print('Training Loss: ', loss.item())
                accuracy = compute_accuracy(predictions, batch_labels)
                print(f'Training Accuracy: {accuracy:.2f}%')

                # Compute validation loss and accuracy.
                evaluate(val_loader, model, loss_fn)

                model.train()

            step += 1

def compute_accuracy(outputs, labels):
    threshold = 0.5
    predictions = (outputs > threshold).float()
    accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
    return accuracy

def evaluate(val_loader, model, loss_fn):
    model.eval()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = model.to(device)
    total_accuracy = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_inputs, batch_labels in val_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            predictions = model(batch_inputs)

            loss = loss_fn(predictions.squeeze(), batch_labels.float())
            accuracy = compute_accuracy(predictions, batch_labels)
            total_accuracy += accuracy
            total_batches += 1

    avg_accuracy = total_accuracy / total_batches
    print(f'Validation Accuracy: {avg_accuracy:.2f}%')
