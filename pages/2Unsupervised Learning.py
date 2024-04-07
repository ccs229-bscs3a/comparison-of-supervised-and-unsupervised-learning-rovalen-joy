#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, metrics
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
import time

# Define the Streamlit app
def app():

    st.subheader('K-means clustering applied to the Wine Quality Dataset')
    text = """This is an example of unsupervised learning task. The WineQT dataset contains 
    information on various physicochemical properties of wines such as 
    alcohol content, acidity levels, and sugar content, but no explicit labels indicating wine quality levels.
    K-means clustering can group wines based on these properties to discover inherent groupings in the data.
    
    \n* **K-means Clustering:** The algorithm aims to partition the data into a predefined number of clusters (k). 
    It assigns each data point to the nearest cluster centroid and recomputes the centroids based on the assigned points. 
    This process minimizes the within-cluster distances, creating groups with similar characteristics. 
    K-means helps uncover inherent groupings within the wine data based on physicochemical properties 
    without relying on predefined quality categories. This allows us to explore how well the data separates 
    into natural clusters, potentially corresponding to different levels of wine quality.

    \n* Choosing the optimal number of clusters (k) is crucial. The "elbow method" 
    helps visualize the trade-off between increasing clusters and decreasing improvement 
    in within-cluster distances.
    * K-means is sensitive to initial centroid placement. Running the algorithm multiple times 
    with different initializations can help identify more stable clusters.
    By applying K-means to the WineQT dataset, we gain insights into the data's underlying structure and can explore
    how well different physicochemical properties delineate the wines into potentially meaningful categories. 
    These clusters might correspond to different qualities or types of wines, providing a basis for further analysis and characterization."""
    st.write(text)


    if st.button("Begin"):
        # Load the Wine Quality dataset
        wine_data = pd.read_csv('WineQT.csv')
        X = wine_data.drop(columns=['quality', 'Id'])  # Features, dropping the columns quality and Id

        # Define the K-means model with 3 clusters (known number of species)
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)

        # Train the K-means model
        kmeans.fit(X)

        # Get the cluster labels for the data
        y_kmeans = kmeans.labels_

        # Since there are no true labels for unsupervised clustering,
        # we cannot directly calculate accuracy.
        # We can use silhouette score to evaluate cluster separation

        # Calculate WCSS
        wcss = kmeans.inertia_
        st.write("Within-Cluster Sum of Squares:", wcss)

        silhouette_score = metrics.silhouette_score(X, y_kmeans)
        st.write("K-means Silhouette Score:", silhouette_score)

        text = """**Within-Cluster Sum of Squares (WCSS): 288310.3478804381 \n
        This value alone doesn't tell the whole story. A lower WCSS generally indicates tighter 
        clusters, but it depends on the scale of your data and the number of clusters used (k).

        \n**K-mmeans Silhouette Score: 0.5269102770307551 \n
        * This score provides a more interpretable measure of cluster quality. It 
        ranges from -1 to 1, where:
        * Values closer to 1 indicate well-separated clusters.
        * Values around 0 suggest clusters are indifferently assigned (data points could belong to either cluster).
        * Negative values indicate poorly separated clusters (data points in a cluster are closer to points in other clusters).
        
        In this case, a Silhouette Score of 0.5269 suggests:
        * **Moderately separated clusters:** The data points within a cluster are somewhat closer to their centroid than to centroids of other clusters. There's some separation, but it's not perfect
        * **Potential for improvement:** You might consider exploring different numbers of clusters (k) or using different initialization methods for K-means to see if a better clustering solution can be achieved with a higher Silhouette Score (closer to 1).
        * The WineQT dataset, with its various physicochemical measurements, offers a good opportunity to test clustering. A Silhouette Score above 0.5 is typically considered decent, but achieving scores closer to 1 would indicate very well-defined clusters.
        * The optimal k can vary depending on the specific dataset and the desired level of granularity in the clustering."""
        with st.expander("Click here for more information."):\
            st.write(text)
            
        # Get predicted cluster labels
        y_pred = kmeans.predict(X)

        # Get unique class labels and color map
        unique_labels = list(set(y_pred))
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(unique_labels)))

        fig, ax = plt.subplots(figsize=(8, 6))

        for label, color in zip(unique_labels, colors):
            indices = y_pred == label
            # Use ax.scatter for consistent plotting on the created axis
            ax.scatter(X.iloc[indices, 10], X.iloc[indices, 1], label=f'Cluster {label}', c=color)

        # Add labels and title using ax methods
        ax.set_xlabel('Alcohol (%)') 
        ax.set_ylabel('Volatile Acidity (g/dm³)')
        ax.set_title('Alcohol vs Volatile Acidity Colored by Predicted Quality')

        # Add legend and grid using ax methods
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


#run the app
if __name__ == "__main__":
    app()
