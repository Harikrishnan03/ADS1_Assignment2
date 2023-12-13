# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 17:00:45 2023

@author: Harikrishnan Marimuthu
"""
# Importing packages:
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    """
    Reads data from a CSV file using pandas.

    Parameters:
    - filename (str): Path to the CSV file.

    Returns:
    - pandas.DataFrame: The loaded data.
    """
    worldbank_df = pd.read_csv(filename, skiprows=4)
    return worldbank_df

def filter_and_transpose_data(worldbank_df, indicator_column, indicator_values,
                              selected_countries, selected_years):
    """
    Filters and transposes data based on specified criteria.

    Parameters:
    - worldbank_df (pandas.DataFrame): Input data.
    - indicator_column (str): Column name for filtering indicators.
    - indicator_values (str or list): Values to filter indicators.
    - selected_countries (list): List of countries to select.
    - selected_years (list): List of years to select.

    Returns:
    - pandas.DataFrame: Filtered and transposed data.
    """
    filtered_data_grouped = worldbank_df.groupby(
        indicator_column, group_keys=True)
    filtered_data = filtered_data_grouped.get_group(indicator_values)
    filtered_data = filtered_data.reset_index()
    filtered_data.set_index('Country Name', inplace=True)
    filtered_data = filtered_data.loc[selected_countries, selected_years]
    filtered_data = filtered_data.dropna(axis=1)
    filtered_data = filtered_data.reset_index()
    transposed_data = filtered_data.set_index(
        'Country Name').transpose().reset_index()
    transposed_data.rename(columns={'index': 'Year'}, inplace=True)  
    return filtered_data, transposed_data

def create_heatmap(worldbank_data, selected_indicators, selected_countries, 
                   selected_years):
    """
    Creates a correlation heatmap for selected indicators.

    Parameters:
    - worldbank_data (pandas.DataFrame): Input data.
    - selected_indicators (list): List of indicators to include in the heatmap.
    - selected_countries (list): List of countries to include.
    - selected_years (list): List of years to include.
    """
    filtered_data_combined = pd.DataFrame()
    for selected_indicator in selected_indicators:
        filtered_data_df, transposed_data_df = filter_and_transpose_data(
            worldbank_data, 'Indicator Name', selected_indicator, 
            selected_countries, selected_years)
        filtered_data_combined[selected_indicator] = transposed_data_df[
            selected_countries].mean(axis=1)
    correlation_matrix = filtered_data_combined.corr()
    plt.figure(figsize=(7, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt='.2f',
                linewidths=0.5)
    plt.title('Correlation Heatmap for Selected Indicators', fontsize=16)
    plt.savefig('heatmap.png')
    plt.show()

def create_line_plot(transposed_data_df, selected_countries,
                     selected_indicators):
    """
    Creates a line plot for selected indicators over the years.

    Parameters:
    - transposed_data_df (pandas.DataFrame): Transposed data.
    - selected_countries (list): List of countries to include.
    - selected_indicators (str): Indicator to plot.
    """
    plt.figure(figsize=(7, 5))
    for country in selected_countries:
        plt.plot(transposed_data_df['Year'], transposed_data_df[country],
                 label=country, linewidth=2)
    plt.title(f'{selected_indicators} for Selected Countries', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel(selected_indicators, fontsize=14)
    plt.legend(fontsize=12, title='Countries', loc='best', title_fontsize='14')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lineplot.png')
    plt.show()

def create_bar_plot(transposed_data_df, selected_countries,
                    selected_indicators):
    """
    Creates a bar plot for selected indicators over the years.

    Parameters:
    - transposed_data_df (pandas.DataFrame): Transposed data.
    - selected_countries (list): List of countries to include.
    - selected_indicators (str): Indicator to plot.
    """
    plt.figure(figsize=(7, 6))
    width = 0.06
    for i, year in enumerate(transposed_data_df['Year']):
        positions = np.arange(len(selected_countries)) + i * width
        plt.bar(positions, transposed_data_df.loc[i, selected_countries],
                width=width, label=str(year), alpha=0.7)
    plt.title(f'{selected_indicators} for Selected Years', fontsize=16)
    plt.xlabel('Country', fontsize=14)
    plt.ylabel('kt of CO2 equivqlent', fontsize=14)
    plt.legend(fontsize=12, title='Years', loc='best', title_fontsize='14')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(np.arange(len(selected_countries)) + width * (
        len(transposed_data_df['Year']) - 1) / 2, selected_countries,
        rotation=30)
    plt.tight_layout()
    plt.savefig('barplot.png')
    plt.show()

def create_histogram_plot(transposed_data_df, selected_countries,
                          selected_indicator):
    """
    Creates a histogram plot for a selected indicator.

    Parameters:
    - transposed_data_df (pandas.DataFrame): Transposed data.
    - selected_countries (str): Country to include.
    - selected_indicator (str): Indicator to plot.
    """
    plt.figure(figsize=(6, 4))
    if selected_countries in transposed_data_df.columns:
        plt.hist(transposed_data_df[selected_countries], bins=20, alpha=0.7,
                 color='purple', edgecolor='black', label=selected_countries)
       
        # Add labels and legend
        plt.title(f'Histogram of {selected_countries} {selected_indicator}',
                  fontsize=13)
        plt.xlabel(selected_indicator, fontsize=11)
        plt.ylabel('Frequency', fontsize=11)
        plt.legend(fontsize=11, title='Country', loc='best',
                   title_fontsize='13')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('histogram.png')
        plt.show()
    else:
        print(f"Selected country '{selected_countries}' not found in data.")

def stat_summary(transposed_data_df):
    """
    Computes numerical summary, skewness, and kurtosis for the given data.

    Parameters:
    - transposed_data_df (pandas.DataFrame): Transposed data.

    Returns:
    - tuple: Numerical summary, skewness, and kurtosis.
    """
    # Use of Describe function:
    numerical_summary = transposed_data_df.describe()
    skewness = transposed_data_df.skew()
    kurt = transposed_data_df.kurtosis()
    return numerical_summary, skewness, kurt

# Select the indicators, countries, and years:
selected_indicators = [
    'Access to electricity (% of population)',
    'Methane emissions (kt of CO2 equivalent)',
    'CO2 emissions from liquid fuel consumption (kt)',
    'Cereal yield (kg per hectare)',
    'Agricultural land (sq. km)'
]
selected_countries = ['North America', 'United States',
                      'Africa Eastern and Southern', 'South Asia', 'Australia']
selected_years = ['2000', '2005', '2010', '2015', '2020']

# Path to the CSV file:
filename = r"C:\Users\hm23abh\Desktop\API_19_DS2_en_csv_v2_6224512.csv"

# Read the data from the CSV file using the defined function:
worldbank_data = read_data(filename)

# Create a heatmap to visualize the correlation between selected indicators:
create_heatmap(worldbank_data, selected_indicators, selected_countries,
               selected_years)

# Filter and transpose data for a specific indicator and create a line plot:
filtered_data_df, transposed_data_df = filter_and_transpose_data(
    worldbank_data, 'Indicator Name',
    selected_indicators[3], selected_countries, selected_years)
create_line_plot(transposed_data_df, selected_countries,
                 selected_indicators[3])

# Filter and transpose data for another indicator and create a line plot:
filtered_data_df, transposed_data_df = filter_and_transpose_data(
    worldbank_data, 'Indicator Name', selected_indicators[4],
    selected_countries, selected_years)
create_line_plot(transposed_data_df, selected_countries,
                 selected_indicators[4])

# Filter and transpose data for a different indicator and create a bar plot:
filtered_data_df, transposed_data_df = filter_and_transpose_data(
    worldbank_data, 'Indicator Name', selected_indicators[1],
    selected_countries, selected_years)
create_bar_plot(transposed_data_df, selected_countries,
                selected_indicators[1])

# Filter and transpose data for another indicator and create a bar plot:
filtered_data_df, transposed_data_df = filter_and_transpose_data(
    worldbank_data, 'Indicator Name', selected_indicators[2],
    selected_countries, selected_years)
create_bar_plot(transposed_data_df, selected_countries,
                selected_indicators[2])

# Create a histogram plot for a specific indicator in a specific region:
create_histogram_plot(transposed_data_df, 'South Asia', selected_indicators[0])

# Compute numerical summary, skewness, and kurtosis:
numerical_summary, skewness, kurt = stat_summary(transposed_data_df)
print("Numerical Summary:")
print(numerical_summary)
print("\nSkewness:")
print(skewness)
print("\nKurtosis:")
print(kurt)