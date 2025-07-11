# LIBRARIES
import pandas as pd
import streamlit as st
import os
from PIL import Image
from data_clean import clean_data, remove_duplicates, engineer_features
from eda import perform_eda, perform_time_series_analysis
from model import run_models
import warnings
warnings.filterwarnings('ignore')

# Streamlit page configuration
st.set_page_config(page_title="LA Crime Analysis", layout="wide")

def load_data(url=None, file_path=None):
    """Load the LA crime dataset from URL or local file."""
    try:
        if url:
            df = pd.read_csv(url)
        elif file_path:
            df = pd.read_csv(file_path)
        df.to_csv('la_crime_data.csv', index=False)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def run_streamlit_app():
    """Main Streamlit application"""
    st.title("Los Angeles Crime Data Analysis (2020-Present)")
    
    # Sidebar controls
    data_source = st.sidebar.radio("Choose data source:", ["Live Data.gov API", "Upload Local File"])
    
    if data_source == "Live Data.gov API":
        url = "https://data.lacity.org/api/views/2nrs-mtv8/rows.csv?accessType=DOWNLOAD"
        df = load_data(url=url)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
        df = load_data(file_path=uploaded_file) if uploaded_file else None

    if df is not None:
        with st.spinner('Processing data...'):
            
            # Display processed data overview
            tab1, tab2, tab3 = st.tabs(["Data Overview", "EDA Visualizations", "Model Results"])
            
            with tab1:
                # Data processing pipeline
                df_clean = clean_data(df)
                df_no_dupes = remove_duplicates(df_clean)
                df_featured = engineer_features(df_no_dupes)

                # Display raw data
                st.subheader("Raw Data Overview")
                st.dataframe(df_featured.head(), use_container_width=True)
                st.subheader("Processed Data Overview")
                st.dataframe(df_featured.head())
                st.subheader("Dataset Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Basic Statistics**")
                    st.write(df_featured.describe())
                with col2:
                    st.write("**Missing Values Summary**")
                    st.write(df_featured.isna().sum().to_frame('Missing Values'))

            with tab2:
                st.subheader("Exploratory Data Analysis")
                eda_figs = perform_eda(df_featured)
                for fig_path in eda_figs:
                    if fig_path.endswith('.png'):
                        st.image(fig_path, use_container_width=True)
                    elif fig_path.endswith('.html'):
                        with open(fig_path, 'r') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=600)
                
                st.subheader("Time Series Analysis")
                ts_figs = perform_time_series_analysis(df_featured)
                for fig_path in ts_figs:
                    if fig_path.endswith('.png'):
                        st.image(fig_path, use_container_width=True)
                    elif fig_path.endswith('.html'):
                        with open(fig_path, 'r') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=600)

            with tab3:
                st.subheader("Machine Learning Models Output")
                dt_results, cluster_labels = run_models(df_featured)
                
                # Model results columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Cluster Analysis")
                    st.write("Crime Cluster Distribution:")
                    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
                    st.bar_chart(cluster_counts)
                    
                with col2:                
                    st.markdown("### Decision Tree Classification")
                    st.dataframe(pd.DataFrame.from_dict(dt_results, orient='index', columns=['Value']))
        # Save and show processed data
        if st.button('Show Processed Data'):
            st.write(df_featured)
            
        st.success("Analysis complete!")

if __name__ == "__main__":
    run_streamlit_app()
