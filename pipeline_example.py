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
    train = data[data['userid'].isin(train_users)]
    val = data[data['userid'].isin(val_users)]
    test = data[data['userid'].isin(test_users)]

    # Check if each split has all event types
    if not (has_all_event_types(train) and has_all_event_types(val) and has_all_event_types(test)):
        print("Warning: Not all splits contain all event types. Consider adjusting the split or handling this case.")

    # Sort each split by user and timestamp
    train = train.sort_values(['userid', 'timestamp'])
    val = val.sort_values(['userid', 'timestamp'])
    test = test.sort_values(['userid', 'timestamp'])

    return train, val, test


# Load your data
data = pd.read_csv('/content/train_dataset.csv')

# Perform the split
train, val, test = user_based_splitting(data)

# Print some statistics
print(f"Total users: {data['userid'].nunique()}")
print(f"Train users: {train['userid'].nunique()}")
print(f"Validation users: {val['userid'].nunique()}")
print(f"Test users: {test['userid'].nunique()}")

print("\nEvent type distribution:")
print("Train:", train['event_type'].value_counts(normalize=True))
print("Validation:", val['event_type'].value_counts(normalize=True))
print("Test:", test['event_type'].value_counts(normalize=True))


import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# Assume train, val, and test are already DataFrame objects with your split data

# Create mappings
user_to_index = {uid: idx for idx, uid in enumerate(pd.concat([train, val, test])['userid'].unique())}
item_to_index = {iid: idx for idx, iid in enumerate(pd.concat([train, val, test])['item_id'].unique())}
category_to_index = {cid: idx for idx, cid in enumerate(pd.concat([train, val, test])['category'].unique())}
event_to_index = {'anchor': 0, 'click': 1, 'purchase': 2}

# Reverse mappings for recommendations
index_to_item = {v: k for k, v in item_to_index.items()}

def create_graph(df):
    num_users = len(user_to_index)
    num_items = len(item_to_index)
    num_categories = len(category_to_index)
    
    edge_index = []
    edge_type = []
    
    for _, row in df.iterrows():
        user_idx = user_to_index[row['userid']]
        item_idx = item_to_index[row['item_id']] + num_users
        edge_index.append([user_idx, item_idx])
        edge_type.append(event_to_index[row['event_type']])
        
        category_idx = category_to_index[row['category']] + num_users + num_items
        edge_index.append([item_idx, category_idx])
        edge_type.append(3)  # item-category relationship
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    
    # Create node features (you might want to use more meaningful features in practice)
    x = torch.randn((num_users + num_items + num_categories, 16))
    
    return Data(x=x, edge_index=edge_index, edge_type=edge_type)

train_graph = create_graph(train)
val_graph = create_graph(val)
test_graph = create_graph(test)



# GAT-based Recommendation Model


class RecommendationGAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, heads=4):
        super(RecommendationGAT, self).__init__()
        self.gat1 = GATConv(num_features, hidden_channels, heads=heads, dropout=0.6)
        self.gat2 = GATConv(hidden_channels * heads, num_classes, heads=1, concat=False, dropout=0.6)
        self.edge_embedding = torch.nn.Embedding(4, num_features)  # 4 types: anchor, click, purchase, item-category

    def forward(self, x, edge_index, edge_type):
        edge_attr = self.edge_embedding(edge_type)
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.gat1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gat2(x, edge_index, edge_attr)
        return x

model = RecommendationGAT(num_features=16, hidden_channels=8, num_classes=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(train_graph.x, train_graph.edge_index, train_graph.edge_type)
    # Only consider user-item edges for loss calculation
    user_item_mask = train_graph.edge_type < 3
    target = F.one_hot(train_graph.edge_type[user_item_mask], num_classes=3).float()
    loss = F.cross_entropy(out[train_graph.edge_index[0]][user_item_mask], target)
    loss.backward()
    optimizer.step()
    return loss

def evaluate(graph):
    model.eval()
    with torch.no_grad():
        out = model(graph.x, graph.edge_index, graph.edge_type)
        # Only consider user-item edges for evaluation
        user_item_mask = graph.edge_type < 3
        pred = out[graph.edge_index[0]][user_item_mask].argmax(dim=1)
        target = graph.edge_type[user_item_mask]
        accuracy = (pred == target).float().mean()
    return accuracy


# Training loop
for epoch in range(300):
    loss = train()
    train_acc = evaluate(train_graph)
    val_acc = evaluate(val_graph)
    if epoch % 20 == 0:
        print(f'Epoch: {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

# Final evaluation
test_acc = evaluate(test_graph)
print(f'Test Accuracy: {test_acc:.4f}')


import torch
import torch.nn.functional as F

def get_recommendations(user_id, top_k=5):
    if user_id not in user_to_index:
        print(f"User {user_id} not in training set. Cannot make recommendations.")
        return []

    model.eval()
    with torch.no_grad():
        out = model(train_graph.x, train_graph.edge_index, train_graph.edge_type)
    
    user_idx = user_to_index[user_id]
    num_users = len(user_to_index)
    num_items = len(item_to_index)
    
    # Debugging information
    print(f"User index: {user_idx}")
    print(f"Number of users: {num_users}")
    print(f"Number of items: {num_items}")
    print(f"Shape of out tensor: {out.shape}")
    
    # Get all items
    all_items = torch.arange(num_users, num_users + num_items)
    
    # Get predictions for all items and apply softmax
    item_scores = F.softmax(out[all_items], dim=1)[:, 2]  # Index 2 for purchase probability
    
    # Debugging information
    print(f"Shape of item_scores: {item_scores.shape}")
    print(f"Top 5 item scores: {item_scores[:5]}")
    
    # Sort items by purchase probability
    sorted_items = torch.argsort(item_scores, descending=True)
    
    # Debugging information
    print(f"Sorted items: {sorted_items[:5]}")
    
    # Get top-k recommendations
    recommendations = []
    for idx in sorted_items[:top_k]:
        item_index = idx.item() - num_users
        item_id = index_to_item[idx.item()]
        score = item_scores[idx].item()  # Correct score fetching
        recommendations.append((item_id, score))
        if len(recommendations) == top_k:
            break
    
    if not recommendations:
        print("No recommendations were generated. Debugging information:")
        print(f"User index: {user_idx}")
        print(f"Number of users: {num_users}")
        print(f"Number of items: {num_items}")
        print(f"Shape of out tensor: {out.shape}")
        print(f"Shape of item_scores: {item_scores.shape}")
        print(f"Top 5 item scores: {item_scores[:5]}")
    
    return recommendations

# Example usage
user_id = 1014  # Example user
recommended_items = get_recommendations(user_id, top_k=5)
if recommended_items:
    print(f"Top 5 recommended items for user {user_id}:")
    for item, score in recommended_items:
        print(f"Item {item}: Score {score:.4f}")
else:
    print(f"No recommendations for user {user_id}")
