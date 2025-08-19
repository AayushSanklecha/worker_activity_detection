import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define activity categories based on the actual dataset
idle_classes = [
    "writing_on_a_book", "reading", "drinking", "sleeping", 
    "talking_on_phone", "smoking", "brushing_teeth"
]

active_classes = [
    "fixing_a_car", "playing_guitar", "cleaning_floor",
    "cooking", "riding_bike", "running", "cutting_vegetables",
    "using_computer", "hammering", "pouring_liquid"
]

def load_data(dataset_dir, img_size=128):
    X, y = [], []
    
    # Check if directory exists
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} not found!")
        return np.array(X), np.array(y)
    
    # Get all image files in the directory
    image_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No image files found in {dataset_dir}")
        return np.array(X), np.array(y)
    
    print(f"Found {len(image_files)} image files")
    
    for img_file in image_files:
        # Extract activity from filename (assuming format: activity_name_number.jpg)
        activity_name = img_file.split('_')[0]  # Get first part of filename
        
        label = None
        if any(idle_class in img_file.lower() for idle_class in idle_classes):
            label = 0
        elif any(active_class in img_file.lower() for active_class in active_classes):
            label = 1
        else:
            continue  # skip classes we don't care about
        
        img_path = os.path.join(dataset_dir, img_file)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(label)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue
    
    print(f"Successfully processed {len(X)} images")
    return np.array(X), np.array(y)

if __name__ == "__main__":
    dataset_path = "dataset/Standford40/JPEGImages"
    X, y = load_data(dataset_path, img_size=128)
    print("Dataset shape:", X.shape, y.shape)
    
    if len(X) > 0:
        np.savez("dataset/stanford40_idle_active.npz", X=X, y=y)
        print(f"Saved dataset with {len(X)} images")
    else:
        print("No valid images found to save")
