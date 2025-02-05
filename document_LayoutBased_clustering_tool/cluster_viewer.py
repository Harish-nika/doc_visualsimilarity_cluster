import pandas as pd
import os
import cv2
from matplotlib import pyplot as plt

# Paths
excel_path = "/home/harish/document_LayoutBased_clustering_tool/cluster_information/image_clusters_kmeans.xlsx"
image_folder = "/home/harish/document_LayoutBased_clustering_tool/data_images/test1_datas"

# Read the Excel file
df = pd.read_excel(excel_path)

# Function to display images from a specific cluster
def show_images_from_cluster(cluster_id, num_images=5):
    """Display images belonging to a specific cluster."""
    # Filter images belonging to the requested cluster
    cluster_images = df[df['cluster'] == cluster_id]['filename'].tolist()

    # Select only the first 'num_images' images
    cluster_images = cluster_images[:num_images]

    if not cluster_images:
        print(f"No images found for Cluster {cluster_id}!")
        return
    
    # Display images
    fig, axes = plt.subplots(1, len(cluster_images), figsize=(15, 5))
    if len(cluster_images) == 1:
        axes = [axes]  # Convert to iterable if only one image

    for ax, image_name in zip(axes, cluster_images):
        image_path = os.path.join(image_folder, image_name)
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib
            ax.imshow(img)
            ax.set_title(image_name)
            ax.axis("off")
        else:
            print(f" Image not found: {image_path}")

    plt.show()

# Example: Show images from Cluster 11
show_images_from_cluster(cluster_id=20, num_images=5)


#http://192.168.1.220/harish.senthil/doc_clust.git 