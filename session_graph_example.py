def remap_item_ids(dataset):
    unique_items = set()
    for data in dataset:
        unique_items.update(data.x.squeeze().tolist())
    
    id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_items))}
    reverse_id_map = {new_id: old_id for old_id, new_id in id_map.items()}
    
    for data in dataset:
        data.x = torch.tensor([[id_map[id.item()]] for id in data.x], dtype=torch.long)
    
    return len(id_map), id_map, reverse_id_map

train_dataset = GraphDataset('./', 'train')
val_dataset = GraphDataset('./', 'val')
test_dataset = GraphDataset('./', 'test')

num_items, id_map, reverse_id_map = remap_item_ids(train_dataset)
remap_item_ids(val_dataset)
remap_item_ids(test_dataset)

print(f"New number of unique items: {num_items}")
