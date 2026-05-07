import pandas as pd
import os

# Print current working directory (important for debugging)
print("Current working directory:", os.getcwd())


# Option 2: If file is in same folder as script, use:
# file_path = "house_data.csv"

# Check if file exists before loading
if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    print("\nFile loaded successfully!\n")
    print(data.head())  # show first 5 rows
else:
    print("\nERROR: File not found!")
    print("Please check the path:", file_path)
    print("Also ensure the file name is correct.")