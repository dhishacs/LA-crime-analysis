import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


def prepare_data(df):
    features = ['AREA', 'Vict Age', 'Vict Sex', 'Vict Descent', 
                'Premis Cd', 'Weapon Used Cd']
    target = 'Crm Cd'  # Crime Code

    # Handle missing values
    df = df[features + [target]].dropna()
    
    # Encode categorical variables
    X = pd.get_dummies(df[features], drop_first=True)
    y = df[target]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)


def decision_tree_model(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    # Print predictions in terminal (optional)
    print("Predictions:")
    print(y_pred)

    """# Visualize the decision tree
    st.subheader("Decision Tree Visualization")
    fig, ax = plt.subplots(figsize=(20, 10))  # Adjust size as needed
    plot_tree(dt, filled=True, feature_names=X_train.columns, class_names=True, max_depth=3, fontsize=10)
    st.pyplot(fig)
    plt.close()"""

    return evaluate_classification(y_test, y_pred)

def kmeans_clustering(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    return cluster_labels


def evaluate_classification(y_true, y_pred):
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }


def evaluate_regression(y_true, y_pred):
    return {
        'mse': np.mean((y_true - y_pred) ** 2),
        'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2)),
        'mae': np.mean(np.abs(y_true - y_pred))
    }


def agnes_streamlit(df):
    """AGNES clustering with Streamlit components"""
    required_cols = ['LAT', 'LON', 'Crm Cd', 'Crm Cd Desc']
    if not all(col in df.columns for col in required_cols):
        st.error("Missing required columns for AGNES clustering")
        return None

    # Main expander block
    with st.expander("AGNES Hierarchical Clustering Analysis"):
        st.header("Geospatial Crime Pattern Analysis")

        # User controls
        sample_size = st.slider("Sample size for AGNES", 500, 5000, 1000, 500)
        max_distance = st.slider("Cluster distance threshold", 0.01, 1.0, 0.1, 0.01)

        # Sample and normalize
        df_sample = df.sample(n=sample_size, random_state=42)
        scaler = MinMaxScaler()
        df_sample[['LAT', 'LON', 'Crm Cd']] = scaler.fit_transform(
            df_sample[['LAT', 'LON', 'Crm Cd']]
        )

        # Perform clustering
        with st.spinner('Performing hierarchical clustering...'):
            Z = linkage(df_sample[['LAT', 'LON', 'Crm Cd']],
                        method='average', metric='euclidean')

            # Dendrogram plot
            fig, ax = plt.subplots(figsize=(10, 7))
            dendrogram(Z, truncate_mode='level', p=5, ax=ax)
            ax.set_title('Dendrogram of Crime Pattern Clustering')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Distance')
            st.pyplot(fig)
            plt.close()

            # Assign clusters
            clusters = fcluster(Z, max_distance, criterion='distance')
            df_sample['Cluster'] = clusters

        # Cluster distribution
        st.subheader("Cluster Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Cluster Distribution**")
            cluster_counts = df_sample['Cluster'].value_counts()
            st.bar_chart(cluster_counts)

        with col2:
            st.write("**Cluster Characteristics** (see below for details)")

    # OUTSIDE MAIN EXPANDER: Show per-cluster analysis
    cluster_counts = df_sample['Cluster'].value_counts()
    for cluster in cluster_counts.index[:5]:  # Limit to first 5 clusters
        cluster_data = df_sample[df_sample['Cluster'] == cluster]
        top_crimes = cluster_data['Crm Cd Desc'].value_counts().head(3)

        with st.expander(f"Cluster {cluster} (Size: {len(cluster_data)})"):
            st.write("Most frequent crime types:")
            for crime, count in top_crimes.items():
                st.write(f"- {crime}: {count} incidents")

    return df_sample


def run_models(df):
    X_train, X_test, y_train, y_test = prepare_data(df)
    st.subheader("Data:")
    st.write(df.head(5))
    st.write("Training set shape:", X_train.shape)
    st.write("Testing set shape:", X_test.shape)
    # AGNES analysis via Streamlit
    agnes_sample = agnes_streamlit(df)
    dt_results = decision_tree_model(X_train, X_test, y_train, y_test)
    
    for metric, value in dt_results.items():
        print(f"{metric}: {value:.4f}")

    st.subheader("\nK-Means Clustering:")
    cluster_labels = kmeans_clustering(X_train)
    st.write(f"Number of samples in each cluster: {np.bincount(cluster_labels)}")

    return dt_results, cluster_labels