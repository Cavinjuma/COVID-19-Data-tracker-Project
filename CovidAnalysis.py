import pandas as pd

# Load the dataset
covidData = (r"C:\Users\Tjay\OneDrive\Documentos\Datasets\Surveillance.csv")
df = pd.read_csv(covidData)

# Display the first few rows
print("...FIRST 5 ROWS...")
print(df.head())
print("...LAST 5 ROWS...")
print(df.tail())

# Check data types and non-null counts
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Convert '+' to 1 and '-' to 0
binary_columns = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07']
df[binary_columns] = df[binary_columns].applymap(lambda x: 1 if x == '+' else 0)

# Compute basic statistics
print(df[binary_columns].describe())

# Group by 'Categories' and compute mean of binary features
grouped_means = df.groupby('Categories')[binary_columns].mean()
print(grouped_means)

import matplotlib.pyplot as plt
import seaborn as sns

# Set the style
sns.set(style="whitegrid")

# Transpose for plotting
grouped_means.T.plot(kind='line', figsize=(10,6))
plt.title('Mean Symptom Presence Across Categories')
plt.xlabel('Symptoms')
plt.ylabel('Mean Presence')
plt.legend(title='Category')
plt.show()


# Plot bar chart
grouped_means.plot(kind='bar', figsize=(10,6))
plt.title('Average Symptom Presence per Category')
plt.xlabel('Category')
plt.ylabel('Average Presence')
plt.legend(title='Symptoms')
plt.show()


# Plot histogram for A01
sns.histplot(df['A01'], bins=2, kde=False)
plt.title('Distribution of Symptom A01')
plt.xlabel('Presence')
plt.ylabel('Count')
plt.show()


# Scatter plot between A01 and A02
sns.scatterplot(x='A01', y='A02', hue='Categories', data=df)
plt.title('A01 vs A02 by Category')
plt.xlabel('A01')
plt.ylabel('A02')
plt.show()
