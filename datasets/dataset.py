import pandas as pd

# Load the CSV file
df = pd.read_csv("data.csv")

# Check structure and contents
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())
print("\nData Info:\n")
df.info()
