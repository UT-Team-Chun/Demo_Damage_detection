import random
import numpy as np
import matplotlib.pyplot as plt

def visualize_random_sample(train_dataset, test_dataset):
    # Visualize a random training sample
    if len(train_dataset) > 0:
        train_idx = random.randint(0, len(train_dataset) - 1)
        train_img, train_mask = train_dataset[train_idx]
        
        # Denormalize the image for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        train_img = train_img.numpy().transpose((1, 2, 0))  # CHW -> HWC
        train_img = std * train_img + mean  # Denormalize
        train_img = np.clip(train_img, 0, 1)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(train_img)
        plt.title("Training Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(train_mask.squeeze(), cmap='gray')  # Remove single-channel dimension
        plt.title("Training Mask")
        plt.axis('off')
        plt.show()
    else:
        print("Training dataset is empty.")

    # Visualize a random testing sample
    if len(test_dataset) > 0:
        test_idx = random.randint(0, len(test_dataset) - 1)
        test_img, test_mask = test_dataset[test_idx]
        
        # Denormalize the image for display
        test_img = test_img.numpy().transpose((1, 2, 0))  # CHW -> HWC
        test_img = std * test_img + mean  # Denormalize
        test_img = np.clip(test_img, 0, 1)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(test_img)
        plt.title("Testing Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(test_mask.squeeze(), cmap='gray')  # Remove single-channel dimension
        plt.title("Testing Mask")
        plt.axis('off')
        plt.show()
    else:
        print("Testing dataset is empty.")