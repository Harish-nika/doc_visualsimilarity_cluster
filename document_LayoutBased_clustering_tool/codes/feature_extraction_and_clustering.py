import os
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Path to your images
path = "/home/harish/workspace_dc/document_LayoutBased_clustering_tool/data_images" #path to the images

# Load pre-trained VGG16 model + higher level layers
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to the input size of VGG16
    img = preprocess_input(img)  # Preprocess for VGG16
    return img

def extract_features(image_paths):
    features = []
    for image_path in image_paths:
        img = load_and_preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        feature = model.predict(img)
        features.append(feature.flatten())
    return np.array(features)

# Get all image paths
image_paths = [os.path.join(path, img) for img in os.listdir(path) if img.endswith('.png')]

# Extract features
features = extract_features(image_paths)# Determine the number of clusters (you can adjust this)


num_clusters = 30  # Number of clusters to create increase if we have too much different types of layout in the images.
# Perform K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(features)

# Assign cluster labels to images
image_cluster_pairs = list(zip(image_paths, clusters))


# Prepare data for saving
data = {
    "ISIN": [os.path.basename(image_path) for image_path, _ in image_cluster_pairs],
    "Cluster": [cluster for _, cluster in image_cluster_pairs],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to Excel file
excel_filename = "/home/harish/workspace_dc/document_LayoutBased_clustering_tool/cluster_information/image_clusters_kmeans.xlsx"
df.to_excel(excel_filename, index=False)

print(f"Results saved to {excel_filename}")
