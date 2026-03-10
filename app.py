import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

st.title("🛍 Mall Customer Segmentation App")
st.markdown("Customer Segmentation using K-Means and Hierarchical Clustering")

@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    return df

df = load_data()

menu = st.sidebar.selectbox(
    "Select Section",
    ["Dataset Overview", "EDA", "K-Means Clustering", "Hierarchical Clustering"]
)

if menu == "Dataset Overview":
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())


elif menu == "EDA":
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Gender', data=df, ax=ax1)
        ax1.set_title("Gender Distribution")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.histplot(df['Age'], bins=20, kde=True, ax=ax2)
        ax2.set_title("Age Distribution")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        fig3, ax3 = plt.subplots()
        sns.histplot(df['Annual Income (k$)'], bins=20, kde=True, ax=ax3)
        ax3.set_title("Annual Income Distribution")
        st.pyplot(fig3)

    with col4:
        fig4, ax4 = plt.subplots()
        sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True, ax=ax4)
        ax4.set_title("Spending Score Distribution")
        st.pyplot(fig4)

    st.subheader("Income vs Spending Score")
    fig5, ax5 = plt.subplots()
    sns.scatterplot(x='Annual Income (k$)',
                    y='Spending Score (1-100)',
                    data=df,
                    ax=ax5)
    st.pyplot(fig5)


elif menu == "K-Means Clustering":
    st.subheader("K-Means Clustering")

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = st.slider("Select Number of Clusters (K)", 2, 10, 5)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, df['KMeans_Cluster'])

    st.write(f"Silhouette Score: **{round(silhouette, 3)}**")

    fig6, ax6 = plt.subplots()
    sns.scatterplot(x='Annual Income (k$)',
                    y='Spending Score (1-100)',
                    hue='KMeans_Cluster',
                    palette='Set1',
                    data=df,
                    ax=ax6)

    ax6.set_title("K-Means Clustering Result")
    st.pyplot(fig6)

    st.subheader("Cluster Summary")
    st.dataframe(
        df.groupby('KMeans_Cluster')[['Annual Income (k$)',
                                      'Spending Score (1-100)']].mean()
    )


elif menu == "Hierarchical Clustering":
    st.subheader("Hierarchical Clustering")

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = st.slider("Select Number of Clusters", 2, 10, 5)

    hc = AgglomerativeClustering(n_clusters=n_clusters)
    df['HC_Cluster'] = hc.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, df['HC_Cluster'])

    st.write(f"Silhouette Score: **{round(silhouette, 3)}**")

    fig7, ax7 = plt.subplots()
    sns.scatterplot(x='Annual Income (k$)',
                    y='Spending Score (1-100)',
                    hue='HC_Cluster',
                    palette='Set2',
                    data=df,
                    ax=ax7)

    ax7.set_title("Hierarchical Clustering Result")
    st.pyplot(fig7)

    st.subheader("Dendrogram")

    fig8 = plt.figure(figsize=(10,5))
    sch.dendrogram(sch.linkage(X_scaled, method='ward'))
    plt.title("Dendrogram")
    plt.xlabel("Customers")
    plt.ylabel("Distance")
    st.pyplot(fig8)

    st.subheader("Cluster Summary")
    st.dataframe(
        df.groupby('HC_Cluster')[['Annual Income (k$)',
                                  'Spending Score (1-100)']].mean()
    )