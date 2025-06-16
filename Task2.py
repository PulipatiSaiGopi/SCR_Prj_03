# Import libraries
from google.colab import files
uploaded = files.upload()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('train.csv')  # Replace with your dataset path

# Display basic information
print("Dataset Shape:", df.shape)
print("\nColumns and Data Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# ---------------------
# Data Cleaning
# ---------------------

# Drop columns with too many missing values or irrelevant ones
# Add 'Name' to the list of columns to drop as it's not numerical
df.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop rows with any remaining missing values (optional)
df.dropna(inplace=True)

# Convert categorical variables to category type
df['Sex'] = df['Sex'].astype('category')
df['Embarked'] = df['Embarked'].astype('category')

# ---------------------
# Exploratory Data Analysis (EDA)
# ---------------------

# 1. Survival count
sns.countplot(data=df, x='Survived')
plt.title('Survival Count')
plt.show()

# 2. Survival rate by sex
sns.countplot(data=df, x='Sex', hue='Survived')
plt.title('Survival by Sex')
plt.show()

# 3. Age distribution
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# 4. Survival rate by passenger class
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survival by Passenger Class')
plt.show()

# 5. Correlation heatmap
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# 6. Boxplot of Age vs Pclass
sns.boxplot(data=df, x='Pclass', y='Age')
plt.title('Age Distribution by Passenger Class')
plt.show()

# ---------------------
# Summary
# ---------------------
print("\nSummary Statistics:\n", df.describe(include='all'))
print("\nCleaned Data Sample:\n", df.head())
