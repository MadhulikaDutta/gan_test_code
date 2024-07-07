import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

# Load and sort data by userid and timestamp to ensure correct sequence processing
data.sort_values(by=['userid', 'timestamp'], inplace=True)

# Filter to include only anchor and click events and then create a deep copy to avoid SettingWithCopyWarning
interaction_data = data[data['event_type'].isin(['anchor', 'click'])].copy()

# Encode user and item IDs
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

interaction_data['user_node'] = user_encoder.fit_transform(interaction_data['userid'])
interaction_data['item_node'] = item_encoder.fit_transform(interaction_data['item_id']) + interaction_data['user_node'].max() + 1

# Prepare labels for next purchase item
interaction_data['next_purchase_item'] = data['item_id'].shift(-1)  # Assume next row is next purchase for simplicity
interaction_data['next_item_node'] = item_encoder.transform(interaction_data['next_purchase_item'].fillna(method='ffill'))

# Convert to numpy arrays before creating torch tensors
user_nodes = interaction_data['user_node'].to_numpy()
item_nodes = interaction_data['item_node'].to_numpy()
edge_index = torch.tensor([user_nodes, item_nodes], dtype=torch.long)

# Node features and labels
num_nodes = interaction_data['item_node'].max() + 1
node_features = torch.eye(num_nodes)
labels = torch.tensor(interaction_data['next_item_node'].to_numpy(), dtype=torch.long)

# Graph Data Object
graph_data = Data(x=node_features, edge_index=edge_index, y=labels)
