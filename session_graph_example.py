import torch
from collections import defaultdict

def remap_item_ids(dataset):
    unique_items = set()
    for data in dataset:
        if isinstance(data.x, torch.Tensor):
            if data.x.numel() == 1:
                unique_items.add(data.x.item())
            else:
                unique_items.update(data.x.view(-1).tolist())
        else:
            unique_items.add(data.x)
    
    id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_items))}
    reverse_id_map = {new_id: old_id for old_id, new_id in id_map.items()}
    
    for data in dataset:
        if isinstance(data.x, torch.Tensor):
            if data.x.numel() == 1:
                data.x = torch.tensor([[id_map[data.x.item()]]], dtype=torch.long)
            else:
                data.x = torch.tensor([[id_map[id.item()]] for id in data.x.view(-1)], dtype=torch.long)
        else:
            data.x = torch.tensor([[id_map[data.x]]], dtype=torch.long)
    
    return len(id_map), id_map, reverse_id_map

try:
    train_dataset = GraphDataset('./', 'train')
    val_dataset = GraphDataset('./', 'val')
    test_dataset = GraphDataset('./', 'test')

    print("Before remapping:")
    print(f"First item in train_dataset: {train_dataset[0].x}")

    num_items, id_map, reverse_id_map = remap_item_ids(train_dataset)
    remap_item_ids(val_dataset)
    remap_item_ids(test_dataset)

    print(f"New number of unique items: {num_items}")
    print("After remapping:")
    print(f"First item in train_dataset: {train_dataset[0].x}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Debugging information:")
    print(f"Type of train_dataset: {type(train_dataset)}")
    print(f"Length of train_dataset: {len(train_dataset)}")
    if len(train_dataset) > 0:
        first_item = train_dataset[0]
        print(f"Type of first item: {type(first_item)}")
        print(f"Type of first item's x attribute: {type(first_item.x)}")
        if isinstance(first_item.x, torch.Tensor):
            print(f"Shape of first item's x attribute: {first_item.x.shape}")
            print(f"Number of elements in first item's x attribute: {first_item.x.numel()}")
        print(f"Value of first item's x attribute: {first_item.x}")

# Print some statistics about the remapped IDs
id_counts = defaultdict(int)
for data in train_dataset:
    id_counts[data.x.item()] += 1

print(f"\nNumber of unique remapped IDs: {len(id_counts)}")
print(f"Most common remapped IDs: {sorted(id_counts.items(), key=lambda x: x[1], reverse=True)[:10]}")
print(f"Least common remapped IDs: {sorted(id_counts.items(), key=lambda x: x[1])[:10]}")
