"""
Le sugiero que ejecute esto en segundo plano.
"""
import random
import sys
from pathlib import Path
from timeit import default_timer as timer  # Measure time

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm  # Import tqdm for progress bar

sys.path.append("../toolbox")
from my_models import FashionMNISTModelV2
from helper_functions import accuracy_fn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 3


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    """
    TRAINING
    Performs training with model trying to learn on data_loader.
    """
    train_loss, train_acc = 0, 0

    # Put model into training mode
    model.train()

    # Loop through the training batches
    for batch, (X, y) in enumerate(data_loader):
        # Put data on target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass (outputs the raw logits from the model)
        y_pred = model(X)

        # 2. Calculate loss and accuracy (per batch)
        loss = loss_fn(y_pred, y)
        train_loss += loss  # accumulate train loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1))  # go from logits -> prediction labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step (update the model's parameters once *per batch*)
        optimizer.step()

    # Divide total train loss and acc by length of train dataloader
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}%")


def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    """
    TESTING
    Performs a testing loop step on model going over data_loader.
    """
    test_loss, test_acc = 0, 0

    # Put the model in eval mode
    model.eval()

    # Turn on inference mode context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send the data to the target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass (outputs raw logits)
            test_pred = model(X)

            # 2. Calculate the loss/acc
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                                    y_pred=test_pred.argmax(dim=1))  # go from logits -> prediction labels

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test acc: {test_acc:.2f}%\n")


def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    # Prints difference between start and end time.
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# EVALUATE MODEL
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device=device):
    """Returns a dictionary containing the results of model predicting on data_loader."""
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            # Make our data device agnostic
            X, y = X.to(device), y.to(device)
            # Make predictions
            y_pred = model(X)

            # Accumulate the loss and acc values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim=1))

        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,  # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}


# MAKE PREDICTIONS
def make_predictions(model: torch.nn.Module,
                     data: list,
                     device: torch.device = device):
    pred_probs = []

    model.to(device)
    model.eval()

    with torch.inference_mode():
        for sample in data:
            # Prepare the sample (add a batch dimension and pass to target device)
            sample = torch.unsqueeze(sample, dim=0).to(device)

            # Forward pass (model outputs raw logits)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            pred_probs.append(pred_prob)

    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)


if __name__ == '__main__':
    # Check versions
    print("Torch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)

    BATCH_SIZE = 32

    # Setup training data
    train_data = datasets.FashionMNIST(
        root="data",  # where to download data to?
        train=True,  # do we want the training dataset?
        download=True,  # do we want to download yes/no?
        transform=torchvision.transforms.ToTensor(),  # how do we want to transform the data?
        target_transform=None  # how do we want to transform the labels/targets?
    )

    # Note: NO ".to(device)" here.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
        target_transform=None
    )

    class_names = train_data.classes
    print("Class names:", class_names)

    # FashionMNIST input shape is (28, 28, 1) because 1 color channel (grayscale)
    # The value of hidden_units is through experimentation and tuning (not because there are 10 possible answers)
    model = FashionMNISTModelV2(input_shape=1,
                                hidden_units=10,
                                output_shape=len(class_names)).to(device)

    # Turn datasets into iterables (batches)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

    train_time_start_model = timer()

    # TRAIN AND TEST MODEL
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n-------")

        train_step(model=model,
                   data_loader=train_dataloader,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   accuracy_fn=accuracy_fn,
                   device=device)

        test_step(model=model,
                  data_loader=test_dataloader,
                  loss_fn=loss_fn,
                  accuracy_fn=accuracy_fn,
                  device=device)

    train_time_end_model = timer()
    total_train_time_model = print_train_time(start=train_time_start_model,
                                              end=train_time_end_model,
                                              device=device)

    # print? total_train_time_model

    # Get model results
    model_results = eval_model(
        model=model,
        data_loader=test_dataloader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )

    print("\nModel Results:", model_results)

    # MAKE RANDOM DUMMY DATASET
    test_samples = []
    test_labels = []

    for sample, label in random.sample(list(test_data), k=9):
        test_samples.append(sample)
        test_labels.append(label)

    # View the first sample shape
    print(test_samples[0].shape)

    # SHOW 1ST SAMPLE
    import matplotlib.pyplot as plt

    plt.title(class_names[test_labels[0]])
    plt.imshow(test_samples[0].squeeze(), cmap="gist_rainbow")
    plt.show()

    # MAKE PREDICTIONS
    pred_probs = make_predictions(model=model,
                                  data=test_samples)

    # VIEW 1ST TWO
    print(pred_probs[:2])

    # CONVERT PROBABILITIES TO LABELS
    pred_classes = pred_probs.argmax(dim=1)
    print("pred_classes", pred_classes)

    plt.figure(figsize=(9, 9))
    nrows = 3
    ncols = 3

    for i, sample in enumerate(test_samples):
        # Create subplot
        plt.subplot(nrows, ncols, i + 1)

        # Plot the target image
        plt.imshow(sample.squeeze(), cmap="gray")

        # Find the prediction (in text form, e.g "Sandal")
        pred_label = class_names[pred_classes[i]]

        # Get the truth label (in text form)
        truth_label = class_names[test_labels[i]]

        # Create a title for the plot
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"

        # Check for equality between pred and truth and change color of title text
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g")  # green text if prediction same as truth
        else:
            plt.title(title_text, fontsize=10, c="r")

        plt.axis(False)

    plt.show()

    # Create model directory path
    MODEL_PATH = Path("../models")
    MODEL_PATH.mkdir(parents=True,
                     exist_ok=True)

    # Create model save
    MODEL_NAME = "03_pytorch_computer_vision_model_1.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # Save the model state dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

    # # Do some real predictions
    from torchmetrics import ConfusionMatrix
    from mlxtend.plotting import plot_confusion_matrix

    from tqdm.auto import tqdm

    # Make predictions with trained model
    y_preds = []
    model.eval()

    with torch.inference_mode():
        for X, y in tqdm(test_dataloader, desc="Making predictions..."):
            # Send the data and targets to target device
            X, y = X.to(device), y.to(device)
            # Do the forward pass
            y_logit = model(X)
            # Turn predictions from logits -> prediction probabilities -> prediction labels
            y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)
            # Put prediction ON CPU for evaluation!
            y_preds.append(y_pred.cpu())

    # Concatenate list of predictions into a tensor
    # print(y_preds)
    y_pred_tensor = torch.cat(y_preds).to(device)
    # y_pred_tensor

    # Confusion Matrix

    # Setup confusion instance and compare predictions to targets
    confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")

    # Move back to CPU
    y_pred_tensor = torch.cat(y_preds).to("cpu")

    confmat_tensor = confmat(
        preds=y_pred_tensor,
        target=test_data.targets  # Labels
    )

    print(confmat_tensor)

    # Plot the confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),  # matplotlib loves numpy
        class_names=class_names,
        figsize=(10, 7)
    )
    plt.show()

    # Note: Labels are sometimes wrong.

    # PLOT THE 8TH ONE
    plt.title(class_names[test_labels[7]])
    plt.imshow(test_samples[7].squeeze(), cmap="gist_heat")
    plt.show()
