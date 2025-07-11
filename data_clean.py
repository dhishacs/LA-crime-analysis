# LIBRARIES
import pandas as pd
import numpy as np
import calendar
import streamlit as st

# Data Cleaning
def clean_data(df):
    """Clean the LA crime dataset."""
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Dataset Columns")
    st.write(df.columns.tolist())

    st.write("### Dataset Info Before Cleaning")
    st.write(f"Shape: {df.shape}")
    st.write("Missing Values (Top 10 Columns):")
    missing_values = df.isnull().sum().sort_values(ascending=False)[:10]
    st.dataframe(missing_values)

    # Convert date columns to datetime
    date_columns = ['DATE OCC', 'DATE RPTD']
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
                st.success(f"Converted {col} to datetime format")
            except Exception as e:
                st.error(f"Failed to convert {col} to datetime: {e}")

    # Handle location data
    if 'LAT' in df.columns and 'LON' in df.columns:
        df = df[(df['LAT'] != 0) & (df['LON'] != 0)]
        df = df.dropna(subset=['LAT', 'LON'])
        st.success("Cleaned location data")

    # Handle missing values for key columns
    key_categorical_cols = ['AREA NAME', 'Crm Cd Desc', 'Vict Descent', 'Vict Sex', 'Weapon Desc', 'Status Desc']
    df.dropna(subset=key_categorical_cols, inplace=True)

    # Display info after cleaning
    st.write("### Dataset Info After Cleaning")
    st.write(f"Shape: {df.shape}")
    missing_values_after = df[key_categorical_cols].isnull().sum()
    st.dataframe(missing_values_after)

    return df

# Duplicate Removal
def remove_duplicates(df):
    """Remove duplicate entries from the dataset."""
    if 'DR_NO' in df.columns:
        duplicates_count = df.duplicated('DR_NO').sum()
        st.write(f"Number of duplicate DR_NO (unique crime ID): {duplicates_count}")

        if duplicates_count > 0:
            df = df.drop_duplicates('DR_NO')
            st.success(f"Removed {duplicates_count} duplicate crime reports.")
            st.write(f"Shape after removing duplicates: {df.shape}")
        else:
            st.info("No duplicates found based on DR_NO.")
    else:
        duplicates_count = df.duplicated().sum()
        st.write(f"Number of completely duplicate rows: {duplicates_count}")

        if duplicates_count > 0:
            df = df.drop_duplicates()
            st.success(f"Removed {duplicates_count} duplicate rows.")
            st.write(f"Shape after removing duplicates: {df.shape}")
        else:
            st.info("No duplicates found.")

    return df

# Feature Engineering
def engineer_features(df):
    """Create useful features from existing data."""
    if 'DATE OCC' in df.columns:
        df['Year'] = df['DATE OCC'].dt.year
        df['Month'] = df['DATE OCC'].dt.month
        df['Month_Name'] = df['Month'].apply(lambda x: calendar.month_abbr[x])
        df['Day'] = df['DATE OCC'].dt.day
        df['Day_of_Week'] = df['DATE OCC'].dt.day_name()

        if 'Hour' in dir(df['DATE OCC'].dt):
            if df['DATE OCC'].dt.hour.nunique() > 1:
                df['Hour'] = df['DATE OCC'].dt.hour
                bins = [0, 6, 12, 18, 24]
                labels = ['Night (00:00-06:00)', 'Morning (06:00-12:00)', 
                          'Afternoon (12:00-18:00)', 'Evening (18:00-24:00)']
                df['Time_of_Day'] = pd.cut(df['Hour'], bins=bins, labels=labels, include_lowest=True)
                st.success("Created time-based features")

    if 'DATE OCC' in df.columns and 'DATE RPTD' in df.columns:
        df['Reporting_Delay_Days'] = (df['DATE RPTD'] - df['DATE OCC']).dt.days
        st.success("Created reporting delay feature")

    if 'Vict Age' in df.columns:
        bins = [-1, 12, 18, 25, 35, 45, 55, 65, 150]
        labels = ['Child (0-12)', 'Teen (13-18)', 'Young Adult (19-25)', 
                  'Adult (26-35)', 'Middle Age (36-45)', 
                  'Older Adult (46-55)', 'Senior (56-65)', 'Elderly (65+)']
        df['Victim_Age_Group'] = pd.cut(df['Vict Age'], bins=bins, labels=labels)
        st.success("Created victim age group feature")

    if 'Crm Cd Desc' in df.columns:
        violent_keywords = ['assault', 'robbery', 'homicide', 'murder', 'rape', 
                            'wound', 'shot', 'battery']
        property_keywords = ['burglary', 'theft', 'stolen', 
                             'larceny', 'shoplifting', 
                             'vandalism', 'damage']
        drugs_keywords = ['drug', 'narcotic', 
                          'marijuana', 
                          'substance']

        def categorize_crime(desc):
            desc = str(desc).lower()
            if any(keyword in desc for keyword in violent_keywords):
                return "Violent Crime"
            elif any(keyword in desc for keyword in property_keywords):
                return "Property Crime"
            elif any(keyword in desc for keyword in drugs_keywords):
                return "Drug Crime"
            else:
                return "Other"

        df['Crime_Category'] = df['Crm Cd Desc'].apply(categorize_crime)
        st.success("Created simplified crime category feature")

    return df
