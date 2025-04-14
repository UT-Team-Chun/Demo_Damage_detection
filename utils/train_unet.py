import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm # For progress bar display

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """Function to train the model"""
    model.train() # Set model to training mode
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix(loss=loss.item()) # Display current batch loss after progress bar

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    print("Training finished.")
    return model # Return the trained model

def evaluate_model(model, test_loader, criterion, device, epsilon=1e-6):
    """Function to evaluate the model"""
    model.eval() # Set model to evaluation mode
    model.to(device)
    running_loss = 0.0
    total_accuracy = 0.0
    total_iou = 0.0
    num_batches = 0

    with torch.no_grad(): # No gradient calculation during evaluation
        progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device) # Ensure masks are float for calculations

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            # Calculate predictions (apply sigmoid and threshold)
            preds = torch.sigmoid(outputs) > 0.5
            preds = preds.float() # Convert boolean to float (0.0 or 1.0)

            # Calculate Accuracy (pixel-wise)
            correct_pixels = (preds == masks).float().sum()
            total_pixels = masks.numel() # Total number of pixels in the batch
            batch_accuracy = correct_pixels / total_pixels
            total_accuracy += batch_accuracy.item()

            # Calculate IoU (Intersection over Union)
            intersection = (preds * masks).sum()
            union = preds.sum() + masks.sum() - intersection
            batch_iou = (intersection + epsilon) / (union + epsilon) # Add epsilon for stability
            total_iou += batch_iou.item()

            num_batches += 1
            progress_bar.set_postfix(loss=loss.item(), acc=batch_accuracy.item(), iou=batch_iou.item())

    avg_loss = running_loss / len(test_loader.dataset)
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0
    avg_iou = total_iou / num_batches if num_batches > 0 else 0

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {avg_accuracy:.4f}")
    print(f"Test IoU (Jaccard): {avg_iou:.4f}")

    return avg_loss, avg_accuracy, avg_iou
