import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit application title
st.title("Category-based Clustering")

# Path to the uploaded Excel file
file_path = './data/keyword.xlsx'

# Read the Excel file
df = pd.read_excel(file_path)
st.write("Uploaded Data:")
st.write(df)

# Category names
category_names = {
    0: 'Thickness',
    1: 'Price',
    2: 'Material',
    3: 'Body',
    4: 'Season',
    5: 'Color',
    6: 'Washing',
    7: 'Length',
    8: 'Design',
    9: 'Size',
    10: 'Delivery',
    11: 'Quality'
}

# Assign human-readable names to categories
df['category_name'] = df['category'].map(category_names)

# Select number of clusters
num_clusters = st.slider("Select the number of clusters", min_value=2, max_value=10, value=3)

# Perform clustering
if st.button("Run Clustering"):
    # Convert category data to numeric
    df['category_numeric'] = pd.factorize(df['category'])[0]

    # Create and train the K-means model
    kmeans = KMeans(n_clusters=num_clusters)
    df['cluster'] = kmeans.fit_predict(df[['category_numeric']])

    st.write("Clustering Results:")
    st.write(df[['keyword', 'count', 'category_name', 'cluster']])


    # Visualize the distribution of categories within each cluster using a bar plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x='cluster', hue='category_name', data=df, palette='viridis')
    plt.title('Category Distribution within Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    st.pyplot(plt)

    # Visualize the keyword counts within each cluster using a box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cluster', y='count', data=df, palette='viridis')
    plt.title('Keyword Counts within Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    st.pyplot(plt)
