import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_and_preprocess_data(filepath):
    """Load the customer segmentation dataset and scale numerical features using NumPy."""
    df = pd.read_csv(filepath)
    
    # Select numerical features and convert to a NumPy array
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    features_np = numerical_df.to_numpy()
    
    # Remove rows with NaN values using NumPy
    valid_rows_mask = ~np.isnan(features_np).any(axis=1)
    df_clean = df[valid_rows_mask].copy()
    clean_features_np = features_np[valid_rows_mask]
    
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clean_features_np)
    
    return df_clean, scaled_features

def perform_clustering(scaled_features, n_clusters=4):
    """Apply KMeans clustering algorithm on scaled NumPy array."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    return clusters

if __name__ == "__main__":
    file_path = "customer_segmentation_data.csv"
    
    print("Loading and preprocessing data with NumPy...")
    df, scaled_data = load_and_preprocess_data(file_path)
    
    print("Running KMeans clustering...")
    df['Cluster'] = perform_clustering(scaled_data, n_clusters=4)
    
    # Save output
    output_path = "customer_segmentation_clustered.csv"
    df.to_csv(output_path, index=False)
    print(f"Clustering complete! Results saved successfully to {output_path}")
