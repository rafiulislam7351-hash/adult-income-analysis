import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jedi.api.refactoring import inline
%matplotlib inline

# ==============================================================================
# 1. DATA INGESTION & VARIABLE INITIALIZATION
# ==============================================================================

# Load the raw dataset from the specified local path
adult_file = pd.read_csv(r"C:\Users\rafiu\OneDrive\Documents\adult11.csv")

# Define semantic column groupings for selective feature tracking
general_intro_col = ['age', 'gender', 'race', 'relationship', 'marital-status', 'native-country']
workclass_col     = ['workclass', 'education', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'salary', 'native-country']

# ==============================================================================
# 2. OUTLIER REMOVAL & DATA FILTERING
# ==============================================================================
# - Drop extreme labor outliers (hours > 84 or hours < 18) to filter out unrealistic/extreme work schedules
# - Drop education anomalies exceeding standard 16-year university milestones if present
adult_file.drop(adult_file[adult_file['hours-per-week'] > 84].index, inplace=True)
adult_file.drop(adult_file[adult_file['hours-per-week'] < 18].index, inplace=True)
adult_file.drop(adult_file[adult_file['education-num'] > 16].index, inplace=True)

# Display columns to verify DataFrame state post-filtering
adult_file.columns

"""
================================================================================
DATA PREPROCESSING: DEMOGRAPHIC FEATURE CONSOLIDATION & ALIGNMENT
================================================================================
OBJECTIVE:
The raw 'marital-status', 'race', and 'relationship' columns contained redundant
information, extreme class imbalances, and highly collinear categories. This
preprocessing step simplifies the demographic feature space to improve machine
learning model performance and prevent redundant feature patterns (multicollinearity).

1. MARITAL STATUS CONSOLIDATION STRATEGY:
   - 'Married'       -> Merged civilian, military (AF), and spouse-absent categories.
   - 'Never_Married' -> Kept intact as a major baseline demographic control.
   - 'Separated'     -> Bundled 'Divorced', 'Separated', and 'Widowed' legal statuses.

2. RACE CONSOLIDATION STRATEGY:
   - 'White' & 'Black' -> Retained due to significant sample sizes.
   - 'Asian'           -> Renamed from 'Asian-Pac-Islander' for cleaner naming conventions.
   - 'Other'           -> Combined 'Amer-Indian-Eskimo' and 'Other' to eliminate sparse classes.

3. RELATIONSHIP CONSOLIDATION STRATEGY:
   - 'Spouse'         -> Combined 'Husband' and 'Wife' into a gender-neutral household role.
   - 'Child'          -> Maps directly from 'Own-child'.
   - 'Unrelated'      -> Merged 'Not-in-family' and 'Unmarried' into a single baseline.
   - 'Other-relative' -> Retained to preserve extended family unit characteristics.
================================================================================
"""

# Processing Marital Status: Strip white spaces and apply consolidated mapping
adult_file['marital-status'] = adult_file['marital-status'].str.strip()
marital_mapping = {
    'Married-civ-spouse': 'Married',
    'Married-spouse-absent': 'Married',
    'Married-AF-spouse': 'Married',
    'Never-married': 'Never_Married',
    'Divorced': 'Separated',
    'Separated': 'Separated',
    'Widowed': 'Separated'
}
adult_file['marital-status'] = adult_file['marital-status'].replace(marital_mapping)

# Processing Race: Normalize text fields and group sparse minority data populations
adult_file['race'] = adult_file['race'].str.strip()
race_mapping = {
    'White': 'White',
    'Black': 'Black',
    'Asian-Pac-Islander': 'Asian',
    'Amer-Indian-Eskimo': 'Other',
    'Other': 'Other'
}
adult_file['race'] = adult_file['race'].replace(race_mapping)

# Processing Family Relationship Roles: Align mapping with engineered marital groups
adult_file['relationship'] = adult_file['relationship'].str.strip()
relationship_mapping = {
    'Husband': 'Spouse',
    'Wife': 'Spouse',
    'Own-child': 'Child',
    'Other-relative': 'Other-relative',
    'Not-in-family': 'Unrelated',
    'Unmarried': 'Unrelated'
}
adult_file['relationship'] = adult_file['relationship'].replace(relationship_mapping)

"""
================================================================================
DATA PREPROCESSING: WORKCLASS & OCCUPATION FEATURE CONSOLIDATION
================================================================================
OBJECTIVE:
Consolidate 'workclass' and 'occupation' columns to mitigate the curse of
dimensionality (reducing possible state interactions from 135 down to 12). Handles
long-tail rare classes and explicitly tracks missing information indicators ('?').
================================================================================
"""

# Processing Workclass: Combine operational fields into broad macroeconomic job sectors
adult_file['workclass'] = adult_file['workclass'].str.strip()
workclass_mapping = {
    'Private': 'Private',
    'Local-gov': 'Government',
    'State-gov': 'Government',
    'Federal-gov': 'Government',
    'Self-emp-not-inc': 'Self-Employed',
    'Self-emp-inc': 'Self-Employed',
    '?': 'Other_Unknown',
    'Without-pay': 'Other_Unknown',
    'Never-worked': 'Other_Unknown'
}
adult_file['workclass'] = adult_file['workclass'].replace(workclass_mapping)

# Processing Occupation: Group micro-roles into distinct economic functional classifications
adult_file['occupation'] = adult_file['occupation'].str.strip()
occupation_mapping = {
    'Exec-managerial': 'White_Collar',
    'Adm-clerical': 'White_Collar',
    'Sales': 'White_Collar',
    'Prof-specialty': 'Professional_Tech',
    'Tech-support': 'Professional_Tech',
    'Craft-repair': 'Blue_Collar',
    'Machine-op-inspct': 'Blue_Collar',
    'Transport-moving': 'Blue_Collar',
    'Handlers-cleaners': 'Blue_Collar',
    'Farming-fishing': 'Blue_Collar',
    'Other-service': 'Service',
    'Protective-serv': 'Service',
    'Priv-house-serv': 'Service',
    '?': 'Other_Unknown',
    'Armed-Forces': 'Other_Unknown'
}
adult_file['occupation'] = adult_file['occupation'].replace(occupation_mapping)

"""
================================================================================
DATA PREPROCESSING: NATIVE-COUNTRY FEATURE CONSOLIDATION
================================================================================
OBJECTIVE:
Compress high-cardinality country values (41 unique states) down to a binary
signal ('United-States' vs 'Non-US') to clear out extremely sparse data entries
and maintain statistical stability.
================================================================================
"""

adult_file['native-country'] = adult_file['native-country'].str.strip()
adult_file['native-country'] = adult_file['native-country'].apply(
    lambda x: 'United-States' if x == 'United-States' else 'Non-US'
)

# ==============================================================================
# 3. DATA VISUALIZATION ENGINE (EDA PILES 1-8)
# ==============================================================================

# --- VISUALIZATION 1: Overall Target Class Imbalance (Pie Chart) ---
# Quantifies baseline distributions across the target column 'salary'
salary_counts = adult_file['salary'].value_counts()
plt.figure(figsize=(8, 4))
plt.pie(salary_counts, labels=salary_counts.index, autopct='%1.1f%%', colors=['#4CAF50', '#FF9800'], startangle=90)
plt.title('Overall Salary Distribution')
plt.show()

# --- VISUALIZATION 2: Income Distribution Across Demographics (Stacked Bar Charts) ---
# Compares percent-normalized distribution metrics for Gender, Race, and Marriage status
demographics = ['gender', 'race', 'marital-status']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

for i, col in enumerate(demographics):
    crosstab_pct = pd.crosstab(adult_file[col], adult_file['salary'], normalize='index') * 100

    crosstab_pct.plot(
        kind='bar',
        stacked=True,
        ax=axes[i],
        color=['#3498db', '#e74c3c'],
        edgecolor='black'
    )

    axes[i].set_title(f'Salary Distribution by {col.capitalize()}', fontsize=14, fontweight='bold')
    axes[i].set_xlabel(col.capitalize(), fontsize=12)
    axes[i].set_ylabel('Percentage (%)', fontsize=12)
    axes[i].set_ylim(0, 110)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# --- VISUALIZATION 3: Age Density Trajectory By Income Bracket (KDE/Histogram) ---
# Plots age distributions simultaneously to capture intersection periods of high vs low earning years
adult_file['age'] = pd.to_numeric(adult_file['age'], errors='coerce')

plt.figure(figsize=(10, 5))
sns.histplot(data=adult_file, x='age', hue='salary', element='step', stat='density', common_norm=False, kde=True, palette=['#3498db', '#e74c3c'], alpha=0.5)
plt.title('Age Distribution Trajectory by Salary Group', fontsize=14, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(axis='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# --- VISUALIZATION 4: Inter-Correlation Matrix of Numerical Columns (Heatmap) ---
# Identifies structural linearity and multi-collinearity links within numeric values
numeric_cols = ['age', 'education-num', 'hours-per-week', 'capital-gain', 'capital-loss']
for col in numeric_cols:
    adult_file[col] = pd.to_numeric(adult_file[col], errors='coerce')

plt.figure(figsize=(6, 4))
correlation_matrix = adult_file[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# --- VISUALIZATION 5: Educational Track Duration vs Income Threshold (Box Plot) ---
# Gauges median and spread differences of school duration across income groups
adult_file['education-num'] = pd.to_numeric(adult_file['education-num'], errors='coerce')

plt.figure(figsize=(8, 4))
sns.boxplot(data=adult_file, x='salary', y='education-num')
plt.title('Education Level (Years) vs Income Success', fontsize=14, fontweight='bold')
plt.xlabel('Salary Class', fontsize=12)
plt.ylabel('Years of Education (education-num)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# --- VISUALIZATION 6: Weekly Work Hour Commitments Across Occupations (Box Plot) ---
# Ranks sorted distributions of work schedules and commitments grouped by occupational fields
adult_file['hours-per-week'] = pd.to_numeric(adult_file['hours-per-week'], errors='coerce')
order = adult_file.groupby('occupation')['hours-per-week'].median().sort_values(ascending=False).index

plt.figure(figsize=(8, 4))
sns.boxplot(data=adult_file, y='occupation', x='hours-per-week', order=order)
plt.title('Distribution of Hours Worked per Week across Occupations', fontsize=14, fontweight='bold')
plt.xlabel('Hours Worked per Week', fontsize=12)
plt.ylabel('Occupation', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# --- VISUALIZATION 7: Investment Returns Over Life Stages (Scatter Plot) ---
# Tracks investment returns across age ranges, filtering out individuals with $0 capital gains
adult_file['age'] = pd.to_numeric(adult_file['age'], errors='coerce')
adult_file['capital-gain'] = pd.to_numeric(adult_file['capital-gain'], errors='coerce')

filtered_df = adult_file[adult_file['capital-gain'] > 0]

plt.figure(figsize=(8, 4))
sns.scatterplot(data=filtered_df, x='age', y='capital-gain', hue='salary', palette=['#3498db', '#e74c3c'], alpha=0.7, s=50)
plt.title('Capital Gains vs. Age & Wealth Bracket (Gains > 0)', fontsize=14, fontweight='bold')
plt.xlabel('Age', fontsize=12)
plt.ylabel('Capital Gains', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# --- VISUALIZATION 8: Income Percentages by Country Groups (Horizontal Bar Chart) ---
# Compares percent-normalized distribution metrics based on foreign vs domestic native origin
top_countries = adult_file['native-country'].value_counts().head(15).index
filtered_countries_df = adult_file[adult_file['native-country'].isin(top_countries)]

country_salary_pct = pd.crosstab(filtered_countries_df['native-country'], filtered_countries_df['salary'], normalize='index') * 100
country_salary_pct = country_salary_pct.sort_values(by='>50K', ascending=True)

country_salary_pct.plot(kind='barh', stacked=True, figsize=(8, 4), color=['#3498db', '#e74c3c'], edgecolor='black')
plt.title('Percentage of High vs Low Earners by Native Country (Top 15 Countries)', fontsize=14, fontweight='bold')
plt.xlabel('Percentage (%)', fontsize=12)
plt.ylabel('Native Country', fontsize=12)
plt.legend(title='Salary')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
