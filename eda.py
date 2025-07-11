import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import warnings
warnings.filterwarnings('ignore')

# Exploratory Data Analysis
def perform_eda(df):
    # Set a consistent style for plots
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

    # Create a directory for saving plots
    import os
    if not os.path.exists('crime_analysis_plots'):
        os.makedirs('crime_analysis_plots')

    visualizations = []

    # 1. Overall crime trends over time
    if 'DATE OCC' in df.columns and 'Year' in df.columns and 'Month' in df.columns:
        # Monthly crime count
        plt.figure(figsize=(15, 7))
        monthly_crimes = df.groupby(['Year', 'Month']).size().reset_index(name='Count')
        monthly_crimes['Date'] = pd.to_datetime(monthly_crimes[['Year', 'Month']].assign(Day=1))
        monthly_crimes = monthly_crimes.sort_values('Date')

        plt.plot(monthly_crimes['Date'], monthly_crimes['Count'], marker='o', linestyle='-', linewidth=2)
        plt.title('Monthly Crime Trends (2020-Present)', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Number of Crimes', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = 'crime_analysis_plots/monthly_crime_trends.png'
        plt.savefig(plot_path)
        visualizations.append(plot_path)
        plt.close()
        print(f"Created time series analysis plot: {plot_path}")

    # 2. Crime type distribution
    if 'Crm Cd Desc' in df.columns:
        plt.figure(figsize=(15, 10))
        top_crimes = df['Crm Cd Desc'].value_counts().nlargest(15)
        sns.barplot(x=top_crimes.values, y=top_crimes.index)
        plt.title('Top 15 Crime Types', fontsize=16)
        plt.xlabel('Number of Incidents', fontsize=14)
        plt.ylabel('Crime Type', fontsize=14)
        plt.tight_layout()
        plot_path = 'crime_analysis_plots/top_crimes.png'
        plt.savefig(plot_path)
        visualizations.append(plot_path)
        plt.close()
        print(f"Created crime type distribution plot: {plot_path}")

    # 3. Crime by area/neighborhood
    if 'AREA NAME' in df.columns:
        plt.figure(figsize=(15, 10))
        area_counts = df['AREA NAME'].value_counts()
        sns.barplot(x=area_counts.values, y=area_counts.index)
        plt.title('Crime Incidents by Neighborhood', fontsize=16)
        plt.xlabel('Number of Incidents', fontsize=14)
        plt.ylabel('Neighborhood', fontsize=14)
        plt.tight_layout()
        plot_path = 'crime_analysis_plots/crimes_by_area.png'
        plt.savefig(plot_path)
        visualizations.append(plot_path)
        plt.close()
        print(f"Created crime by neighborhood plot: {plot_path}")

    # 4. Crime by day of week and time of day (heatmap)
    if 'Day_of_Week' in df.columns and 'Time_of_Day' in df.columns:
        plt.figure(figsize=(12, 8))
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        time_order = ['Night (00:00-06:00)', 'Morning (06:00-12:00)',
                     'Afternoon (12:00-18:00)', 'Evening (18:00-24:00)']

        # Create cross-tabulation
        heatmap_data = pd.crosstab(df['Day_of_Week'], df['Time_of_Day'])
        # Reorder rows and columns
        heatmap_data = heatmap_data.reindex(day_order)
        heatmap_data = heatmap_data.reindex(columns=time_order)

        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Crime Incidents by Day of Week and Time of Day', fontsize=16)
        plt.tight_layout()
        plot_path = 'crime_analysis_plots/day_time_heatmap.png'
        plt.savefig(plot_path)
        visualizations.append(plot_path)
        plt.close()
        print(f"Created day/time heatmap: {plot_path}")

    # 5. Victim demographics
    demographic_plots = []

    # Age distribution
    if 'Victim_Age_Group' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.countplot(x='Victim_Age_Group', data=df)
        plt.title('Crime Victims by Age Group', fontsize=16)
        plt.xlabel('Age Group', fontsize=14)
        plt.ylabel('Number of Victims', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = 'crime_analysis_plots/victim_age_groups.png'
        plt.savefig(plot_path)
        demographic_plots.append(plot_path)
        plt.close()

    # Sex distribution
    if 'Vict Sex' in df.columns:
        plt.figure(figsize=(8, 8))
        sex_counts = df['Vict Sex'].value_counts()

        # Clean labels mapping
        label_map = {
            'F': 'Female',
            'M': 'Male',
            'X': 'Unknown'
        }
        labels = [label_map.get(label, label) for label in sex_counts.index]

        wedges, texts, autotexts = plt.pie(
            sex_counts.values,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12}
        )

        # Improve label clarity
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_fontsize(12)

        plt.axis('equal')
        plt.title('Victim Sex Distribution', fontsize=16, pad=20)  # Add padding to avoid overlap
        plt.tight_layout()
        plot_path = 'crime_analysis_plots/victim_sex.png'
        plt.savefig(plot_path)
        demographic_plots.append(plot_path)
        plt.close()

    if demographic_plots:
        visualizations.extend(demographic_plots)
        print(f"Created {len(demographic_plots)} victim demographic plots")


    # 6. Weapon Usage (if available)
    if 'Weapon Desc' in df.columns:
        plt.figure(figsize=(15, 10))
        # Filter out 'NONE' or similar values and get top weapons
        weapon_data = df[~df['Weapon Desc'].isin(['NONE', 'None', 'N/A', ''])]
        top_weapons = weapon_data['Weapon Desc'].value_counts().nlargest(10)

        sns.barplot(x=top_weapons.values, y=top_weapons.index)
        plt.title('Top 10 Weapons Used in Crimes', fontsize=16)
        plt.xlabel('Number of Incidents', fontsize=14)
        plt.ylabel('Weapon Type', fontsize=14)
        plt.tight_layout()
        plot_path = 'crime_analysis_plots/top_weapons.png'
        plt.savefig(plot_path)
        visualizations.append(plot_path)
        plt.close()
        print(f"Created weapons analysis plot: {plot_path}")

    # 7. Crime Status/Resolution
    if 'Status Desc' in df.columns:
        plt.figure(figsize=(10, 10))  # Bigger figure for potentially long labels
        status_counts = df['Status Desc'].value_counts()

        wedges, texts, autotexts = plt.pie(
            status_counts.values,
            labels=status_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 11}
        )

        # Adjust label font size
        for text in texts:
            text.set_fontsize(11)
        for autotext in autotexts:
            autotext.set_fontsize(10)

        plt.axis('equal')
        plt.title('Crime Resolution Status', fontsize=16, pad=25)  # Padding prevents overlap with the chart
        plt.tight_layout()
        plot_path = 'crime_analysis_plots/resolution_status.png'
        plt.savefig(plot_path)
        visualizations.append(plot_path)
        plt.close()
        print(f"Created crime resolution status plot: {plot_path}")

    # 8. Reporting delay analysis
    if 'Reporting_Delay_Days' in df.columns:
        plt.figure(figsize=(14, 7))
        # Filter to reasonable delays (some might be data errors)
        delay_data = df[df['Reporting_Delay_Days'] <= 30]  # Look at delays up to a month

        sns.histplot(delay_data['Reporting_Delay_Days'], kde=True, bins=30)
        plt.title('Distribution of Reporting Delays (Days)', fontsize=16)
        plt.xlabel('Delay in Days', fontsize=14)
        plt.ylabel('Number of Incidents', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = 'crime_analysis_plots/reporting_delay.png'
        plt.savefig(plot_path)
        visualizations.append(plot_path)
        plt.close()
        print(f"Created reporting delay analysis plot: {plot_path}")

    # 9. Geographic visualization (basic map)
    if 'LAT' in df.columns and 'LON' in df.columns:
        try:
            # Create a sample of data for mapping (full dataset might be too large)
            map_sample = df.sample(min(5000, len(df)))

            # Create a basic folium map centered on Los Angeles
            m = folium.Map(location=[34.0522, -118.2437], zoom_start=10)

            # Add a heatmap
            heat_data = [[row['LAT'], row['LON']] for index, row in map_sample.iterrows()]
            HeatMap(heat_data).add_to(m)

            # Save the map
            map_path = 'crime_analysis_plots/crime_heatmap.html'
            m.save(map_path)
            visualizations.append(map_path)
            print(f"Created crime heatmap: {map_path}")
        except Exception as e:
            print(f"Error creating geographic visualization: {e}")

    print(f"\nEDA complete. Generated {len(visualizations)} visualizations.")
    return visualizations

# Time Series Analysis
def perform_time_series_analysis(df):
    """Analyze crime trends over time."""
    if 'DATE OCC' not in df.columns:
        print("Cannot perform time series analysis without date information.")
        return []

    visualizations = []

    # Create a directory for saving plots if it doesn't exist
    import os
    if not os.path.exists('crime_analysis_plots'):
        os.makedirs('crime_analysis_plots')

    # 1. Monthly trend by major crime types
    if 'Crime_Category' in df.columns:
        # Group by year, month and crime category
        plt.figure(figsize=(16, 10))

        # Create a datetime column for proper time series plotting
        df_ts = df.copy()
        df_ts['YearMonth'] = pd.to_datetime(df_ts[['Year', 'Month']].assign(Day=1))

        # Group and count
        crime_ts = df_ts.groupby(['YearMonth', 'Crime_Category']).size().reset_index(name='Count')

        # Pivot for plotting
        crime_ts_pivot = crime_ts.pivot(index='YearMonth', columns='Crime_Category', values='Count')

        # Plot
        crime_ts_pivot.plot(figsize=(16, 8), marker='o')
        plt.title('Monthly Crime Trends by Category', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Number of Incidents', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Crime Category', fontsize=12)
        plt.tight_layout()
        plot_path = 'crime_analysis_plots/crime_category_trends.png'
        plt.savefig(plot_path)
        visualizations.append(plot_path)
        plt.close()
        print(f"Created crime category trend plot: {plot_path}")

    # 2. Day of week patterns
    if 'Day_of_Week' in df.columns:
        plt.figure(figsize=(14, 7))
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df['Day_of_Week'].value_counts().reindex(day_order)

        sns.barplot(x=day_counts.index, y=day_counts.values)
        plt.title('Crime Incidents by Day of Week', fontsize=16)
        plt.xlabel('Day of Week', fontsize=14)
        plt.ylabel('Number of Incidents', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = 'crime_analysis_plots/day_of_week_pattern.png'
        plt.savefig(plot_path)
        visualizations.append(plot_path)
        plt.close()
        print(f"Created day of week pattern plot: {plot_path}")

    # 3. Hour of day pattern
    if 'Hour' in df.columns:
        plt.figure(figsize=(15, 7))
        hour_counts = df.groupby('Hour').size()

        sns.lineplot(x=hour_counts.index, y=hour_counts.values, marker='o')
        plt.title('Crime Incidents by Hour of Day', fontsize=16)
        plt.xlabel('Hour (24-hour format)', fontsize=14)
        plt.ylabel('Number of Incidents', fontsize=14)
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = 'crime_analysis_plots/hour_of_day_pattern.png'
        plt.savefig(plot_path)
        visualizations.append(plot_path)
        plt.close()
        print(f"Created hour of day pattern plot: {plot_path}")

    return visualizations