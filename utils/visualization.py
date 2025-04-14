import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import random
import matplotlib.colors as mcolors # Import colors

def visualize_segmentation(model, loader, device, num_samples=5, img_mean=[0.485, 0.456, 0.406], img_std=[0.229, 0.224, 0.225], overlay_alpha=0.4, overlay_color='blue'):
    """
    Visualizes original images, ground truth masks, predicted masks,
    and overlays of images with predicted masks using RGBA blending
    on randomly selected samples from the dataset.

    Args:
        model: The trained segmentation model.
        loader: DataLoader for the dataset to visualize.
        device: The device (cpu or cuda) to run inference on.
        num_samples (int): Number of samples to visualize.
        img_mean (list): Mean used for image normalization.
        img_std (list): Standard deviation used for image normalization.
        overlay_alpha (float): Transparency level for mask overlays (0 to 1).
        overlay_color (str or tuple): Color for the overlay mask.
    """
    if loader is None or loader.dataset is None:
        print("Cannot visualize predictions: Data loader or dataset is not available.")
        return

    dataset = loader.dataset
    dataset_len = len(dataset)

    if dataset_len == 0:
        print("Cannot visualize predictions: Dataset is empty.")
        return

    # Ensure num_samples is not greater than dataset length
    num_samples = min(num_samples, dataset_len)

    # Get random indices
    random_indices = random.sample(range(dataset_len), num_samples)

    model.eval()
    model.to(device)

    # Denormalize parameters
    mean = np.array(img_mean)
    std = np.array(img_std)

    # Get overlay color RGBA (normalized 0-1)
    try:
        overlay_rgb = mcolors.to_rgb(overlay_color)
    except ValueError:
        print(f"Warning: Invalid overlay color '{overlay_color}'. Using blue.")
        overlay_rgb = mcolors.to_rgb('blue')

    plt.figure(figsize=(15, 4 * num_samples))

    for i, idx in enumerate(random_indices):
        # Get individual sample
        img_tensor, mask_tensor = dataset[idx] # Get raw sample (potentially transformed)

        # Add batch dimension and move to device for model input
        img_input = img_tensor.unsqueeze(0).to(device)

        # Get prediction for the single image
        with torch.no_grad():
            output = model(img_input)
            pred = torch.sigmoid(output) > 0.5
            pred = pred.squeeze(0).cpu().numpy() # Remove batch dim and move to cpu

        # Prepare image and mask for display
        img_display = img_tensor.cpu().numpy().transpose((1, 2, 0)) # CHW -> HWC
        img_display = std * img_display + mean # Denormalize
        img_display = np.clip(img_display, 0, 1)

        gt_mask = mask_tensor.cpu().numpy().squeeze() # Remove channel dim
        pred_mask = pred.squeeze().astype(bool) # Remove channel dim, ensure boolean

        # --- Plotting ---
        # Original Image
        ax1 = plt.subplot(num_samples, 4, i * 4 + 1)
        ax1.imshow(img_display)
        ax1.set_title("Original Image")
        ax1.axis('off')

        # Ground Truth Mask
        ax2 = plt.subplot(num_samples, 4, i * 4 + 2)
        ax2.imshow(gt_mask, cmap='gray')
        ax2.set_title("Ground Truth Mask")
        ax2.axis('off')

        # Predicted Mask
        ax3 = plt.subplot(num_samples, 4, i * 4 + 3)
        ax3.imshow(pred_mask, cmap='gray')
        ax3.set_title("Predicted Mask")
        ax3.axis('off')

        # Overlay: Image + Predicted Mask (RGBA method)
        ax4 = plt.subplot(num_samples, 4, i * 4 + 4)
        ax4.imshow(img_display) # Display base image first

        # Create an RGBA overlay image
        h, w = pred_mask.shape
        overlay_rgba = np.zeros((h, w, 4), dtype=float)

        # Set color and alpha only where the prediction mask is True
        overlay_rgba[pred_mask, 0] = overlay_rgb[0] # R
        overlay_rgba[pred_mask, 1] = overlay_rgb[1] # G
        overlay_rgba[pred_mask, 2] = overlay_rgb[2] # B
        overlay_rgba[pred_mask, 3] = overlay_alpha    # Alpha

        # Display the overlay RGBA image on top of the original image
        ax4.imshow(overlay_rgba)
        ax4.set_title("Image + Prediction Overlay")
        ax4.axis('off')


    plt.tight_layout()
    plt.show()

def visualize_random_dataset_sample(dataset, img_mean=[0.485, 0.456, 0.406], img_std=[0.229, 0.224, 0.225]):
    """Visualizes a random sample (image and mask) directly from a dataset."""
    if len(dataset) == 0:
        print("Dataset is empty, cannot visualize sample.")
        return

    idx = random.randint(0, len(dataset) - 1)
    img_tensor, mask_tensor = dataset[idx]

    # Denormalize the image for display
    mean = np.array(img_mean)
    std = np.array(img_std)
    img_display = img_tensor.numpy().transpose((1, 2, 0))  # CHW -> HWC
    img_display = std * img_display + mean
    img_display = np.clip(img_display, 0, 1)

    mask_display = mask_tensor.squeeze().numpy() # Remove channel dim and convert to numpy

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_display)
    plt.title("Random Sample Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask_display, cmap='gray')
    plt.title("Random Sample Mask")
    plt.axis('off')
    plt.show()
