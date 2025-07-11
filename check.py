#dummy file to check the result without streamlit
#LA Crime Data Analysis

import pandas as pd
import streamlit as st
from data_clean import clean_data
from data_clean import remove_duplicates
from data_clean import engineer_features
from eda import perform_eda
from eda import perform_time_series_analysis
from model import run_models
import warnings
warnings.filterwarnings('ignore')

print("hi")

# Load the LA Crime dataset
def load_data(url=None, file_path=None):
    """Load the LA crime dataset from URL or local file."""
    try:
        if file_path:
            # Load from local file
            df = pd.read_csv(file_path)
            df.to_csv('la_crime_data.csv', index=False)
            print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
            return df
        elif url:
            # Load directly from data.gov API
            df = pd.read_csv(url)
            df.to_csv('la_crime_data.csv', index=False)
            print(f"Successfully loaded dataset from URL with {df.shape[0]} rows and {df.shape[1]} columns.")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Main analysis function
def analyze_la_crime_data(url=None, file_path=None):
    """Complete analysis pipeline for LA crime dataset."""
    print("Starting LA Crime Data Analysis...")

    # Load the data
    df = load_data(url, file_path)

    if df is not None:
        # Clean the data
        df_clean = clean_data(df)

        # Remove duplicates
        df_no_dupes = remove_duplicates(df_clean)

        # Engineer features
        df_featured = engineer_features(df_no_dupes)

        # Perform EDA
        eda_visualizations = perform_eda(df_featured)

        # Perform time series analysis
        ts_visualizations = perform_time_series_analysis(df_featured)

        #Model
        dt_results, cluster_labels = run_models(df_featured)

        # Print the results
        print("\nK-Means Clustering Labels:")
        print(cluster_labels)
        
        print("Decision Tree Results:")
        print(dt_results)

        # Save processed dataset
        try:
            df_featured.to_csv('processed_la_crime_data.csv', index=False)
            print("Saved processed dataset to 'processed_la_crime_data.csv'")
        except Exception as e:
            print(f"Error saving processed dataset: {e}")

        all_visualizations = eda_visualizations + ts_visualizations
        print(f"\nAnalysis complete. Generated {len(all_visualizations)} visualizations.")
        return df_featured
    else:
        print("Analysis could not be completed due to data loading error.")
        return None

#DATA --> https://catalog.data.gov/dataset/crime-data-from-2020-to-present

#LOAD DATASET
data = analyze_la_crime_data(url="https://data.lacity.org/api/views/2nrs-mtv8/rows.csv?accessType=DOWNLOAD", file_path="LA_crime_data.csv")