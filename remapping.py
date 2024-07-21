import pandas as pd
import numpy as np
import json

# Create a sample dataframe with 10-digit item_ids
np.random.seed(42)
df = pd.DataFrame({
    'item_id': np.random.randint(1000000000, 9999999999, size=23435)
})

# Create a mapping dictionary
unique_item_ids = df['item_id'].unique()
id_mapping = {str(old_id): new_id for new_id, old_id in enumerate(unique_item_ids, start=1)}

# Create reverse mapping
reverse_mapping = {str(v): k for k, v in id_mapping.items()}

# Apply the mapping to create a new column
df['new_item_id'] = df['item_id'].astype(str).map(id_mapping)

# Display the first few rows of the dataframe
print(df.head())

# Print some statistics
print(f"\nNumber of unique original item_ids: {df['item_id'].nunique()}")
print(f"Number of unique new item_ids: {df['new_item_id'].nunique()}")
print(f"Smallest new item_id: {df['new_item_id'].min()}")
print(f"Largest new item_id: {df['new_item_id'].max()}")

# Save the mapping dictionaries to JSON files
with open('id_mapping.json', 'w') as f:
    json.dump(id_mapping, f)

with open('reverse_mapping.json', 'w') as f:
    json.dump(reverse_mapping, f)

print("\nMapping dictionaries saved to 'id_mapping.json' and 'reverse_mapping.json'")

# Example of how to load and use the mapping later
def load_mapping(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Load the mappings
loaded_id_mapping = load_mapping('id_mapping.json')
loaded_reverse_mapping = load_mapping('reverse_mapping.json')

# Example of remapping
original_id = str(df['item_id'].iloc[0])
new_id = loaded_id_mapping[original_id]
recovered_original_id = loaded_reverse_mapping[str(new_id)]

print(f"\nExample remapping:")
print(f"Original ID: {original_id}")
print(f"New ID: {new_id}")
print(f"Recovered Original ID: {recovered_original_id}")
