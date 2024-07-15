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

# Select column for clustering
category_col = st.selectbox("Select the category column", df.columns)

# Select number of clusters
num_clusters = st.slider("Select the number of clusters", min_value=2, max_value=10, value=3)

# Perform clustering
if st.button("Run Clustering"):
    # Convert category data to numeric
    df['category_numeric'] = pd.factorize(df[category_col])[0]

    # Create and train the K-means model
    kmeans = KMeans(n_clusters=num_clusters)
    df['cluster'] = kmeans.fit_predict(df[['category_numeric']])

    st.write("Clustering Results:")
    st.write(df[['keyword', 'count', 'category', 'cluster']])

    # Visualize the clustering results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df.index, y='category_numeric', hue='cluster', data=df, palette='viridis')
    plt.title('Clustering Results by Category')
    plt.xlabel('Index')
    plt.ylabel(category_col)
    st.pyplot(plt)
