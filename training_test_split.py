import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def user_based_splitting(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Ensure the ratios sum to 1
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0)

    # Get unique users
    users = data['userid'].unique()

    # Split users into train, validation, and test sets
    train_users, temp_users = train_test_split(users, train_size=train_ratio, random_state=42)
    val_users, test_users = train_test_split(temp_users, train_size=val_ratio/(val_ratio + test_ratio), random_state=42)

    # Function to check if a split has all event types
    def has_all_event_types(df):
        return set(df['event_type'].unique()) == set(['anchor', 'click', 'purchase'])

    # Split data based on user assignment
    train_data = data[data['userid'].isin(train_users)]
    val_data = data[data['userid'].isin(val_users)]
    test_data = data[data['userid'].isin(test_users)]

    # Check if each split has all event types
    if not (has_all_event_types(train_data) and has_all_event_types(val_data) and has_all_event_types(test_data)):
        print("Warning: Not all splits contain all event types. Consider adjusting the split or handling this case.")

    # Sort each split by user and timestamp
    train_data = train_data.sort_values(['userid', 'timestamp'])
    val_data = val_data.sort_values(['userid', 'timestamp'])
    test_data = test_data.sort_values(['userid', 'timestamp'])

    return train_data, val_data, test_data

# Example usage
def main():
    # Load your data
    data = pd.read_csv('your_data.csv')

    # Perform the split
    train_data, val_data, test_data = user_based_splitting(data)

    # Print some statistics
    print(f"Total users: {data['userid'].nunique()}")
    print(f"Train users: {train_data['userid'].nunique()}")
    print(f"Validation users: {val_data['userid'].nunique()}")
    print(f"Test users: {test_data['userid'].nunique()}")

    print("\nEvent type distribution:")
    print("Train:", train_data['event_type'].value_counts(normalize=True))
    print("Validation:", val_data['event_type'].value_counts(normalize=True))
    print("Test:", test_data['event_type'].value_counts(normalize=True))

if __name__ == "__main__":
    main()
