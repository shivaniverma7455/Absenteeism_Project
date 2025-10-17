import numpy as np
import matplotlib
import scipy
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
try:
    absentees_data = pd.read_csv('data/Absenteeism_at_work.csv', sep=';')
except FileNotFoundError:
    print("Error: 'Absenteeism_at_work.csv' not found. Please check the data/ directory.")
    exit()

# --- Data Exploration and Preprocessing ---
print("Dataset Head:")
print(absentees_data.head())

print("\nDataset Info:")
absentees_data.info()

print("\nDescriptive Statistics:")
x = absentees_data.describe()
x.to_csv('output/absenteeism_statistics.txt')
print("Stats saved to output/absenteeism_statistics.txt")

# --- Data Visualization ---
x = absentees_data['Age']
y = absentees_data['Distance from Residence to Work']
plt.scatter(x, y, color='green', marker='o', s=10)
plt.ylim(0, 30)
plt.xlim(35, 60)
plt.title('Age vs Distance from Residence to Work')
plt.xlabel('Age')
plt.ylabel('Distance from Residence to Work')
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(absentees_data['Month of absence'], bins=12)
ax.set_title("Absences by Month")
ax.set_xlabel("Month")
plt.show()

monthly_absence = absentees_data.groupby('Month of absence')['Absenteeism time in hours'].mean()
monthly_absence.plot(kind='line')
plt.title('Mean Absenteeism Time by Month')
plt.xlabel('Month')
plt.ylabel('Mean Absenteeism Time (hours)')
plt.show()

figure = plt.figure()
ax = figure.add_subplot(1, 1, 1)
ax.boxplot(absentees_data['Work load Average/day '])
plt.title('Work Load Average per Day')
plt.ylabel('Work load Average/day')
plt.show()

# --- Seasonal Analysis ---
if 'Seasons' in absentees_data.columns:
    season_data = absentees_data.groupby('Seasons').sum()
    x = season_data['Absenteeism time in hours']
    labels = x.index
    plt.pie(x, labels=labels, autopct="%1.1f%%")
    plt.title("Absences by Season")
    plt.axis("equal")
    plt.legend()
    plt.show()

# --- Handle missing values ---
absentees_data.fillna(absentees_data.median(), inplace=True)

# --- Data type conversion ---
if 'Reason for absence' in absentees_data.columns:
    absentees_data['Reason for absence'] = absentees_data['Reason for absence'].astype('category')

# --- Statistical Analysis ---
if 'Absenteeism time in hours' in absentees_data.columns:
    # NOTE: Update 'Work_days' to the actual column name representing total working days
    work_days_col = None
    for col in absentees_data.columns:
        if 'work' in col.lower() and 'days' in col.lower():
            work_days_col = col
            break
    if work_days_col:
        total_working_days = absentees_data[work_days_col].sum()
        absenteeism_rate = (absentees_data['Absenteeism time in hours'].sum() / total_working_days) * 100
        print(f"\nOverall Absenteeism Rate: {absenteeism_rate:.2f}%")

    print("\nAbsenteeism by Reason for absence:")
    print(absentees_data.groupby('Reason for absence')['Absenteeism time in hours'].sum().sort_values(ascending=False))

    print("\nAbsenteeism by Month of absence:")
    print(absentees_data.groupby('Month of absence')['Absenteeism time in hours'].sum())

    print("\nCorrelation Matrix:")
    numerical_cols = absentees_data.select_dtypes(include=['number']).columns
    print(absentees_data[numerical_cols].corr()['Absenteeism time in hours'].sort_values(ascending=False))

    # --- Visualization ---
    plt.figure(figsize=(8, 6))
    sns.histplot(absentees_data['Absenteeism time in hours'], bins=20, kde=True)
    plt.title('Distribution of Absenteeism Time in Hours')
    plt.xlabel('Absenteeism Time (Hours)')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Reason for absence', y='Absenteeism time in hours', data=absentees_data)
    plt.title('Absenteeism Time by Reason for Absence')
    plt.xlabel('Reason for Absence')
    plt.ylabel('Absenteeism Time (Hours)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Pairplot for selected numerical features
    if all(col in absentees_data.columns for col in ['Age', 'Transportation expense', 'Absenteeism time in hours']):
        sns.pairplot(absentees_data[['Age', 'Transportation expense', 'Absenteeism time in hours']])
        plt.show()