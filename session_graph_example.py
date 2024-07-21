def remap_item_ids(dataset):
    unique_items = set()
    for data in dataset:
        if isinstance(data.x, torch.Tensor):
            if data.x.dim() == 2:
                unique_items.update(data.x.squeeze().tolist())
            elif data.x.dim() == 1:
                unique_items.update(data.x.tolist())
            else:
                unique_items.add(data.x.item())
        elif isinstance(data.x, list):
            unique_items.update(data.x)
        else:
            unique_items.add(data.x)
    
    id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_items))}
    reverse_id_map = {new_id: old_id for old_id, new_id in id_map.items()}
    
    for data in dataset:
        if isinstance(data.x, torch.Tensor):
            if data.x.dim() == 2:
                data.x = torch.tensor([[id_map[id.item()]] for id in data.x], dtype=torch.long)
            elif data.x.dim() == 1:
                data.x = torch.tensor([id_map[id.item()] for id in data.x], dtype=torch.long)
            else:
                data.x = torch.tensor([id_map[data.x.item()]], dtype=torch.long)
        elif isinstance(data.x, list):
            data.x = torch.tensor([id_map[id] for id in data.x], dtype=torch.long)
        else:
            data.x = torch.tensor([id_map[data.x]], dtype=torch.long)
    
    return len(id_map), id_map, reverse_id_map

try:
    train_dataset = GraphDataset('./', 'train')
    val_dataset = GraphDataset('./', 'val')
    test_dataset = GraphDataset('./', 'test')

    num_items, id_map, reverse_id_map = remap_item_ids(train_dataset)
    remap_item_ids(val_dataset)
    remap_item_ids(test_dataset)

    print(f"New number of unique items: {num_items}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Debugging information:")
    print(f"Type of train_dataset: {type(train_dataset)}")
    print(f"Length of train_dataset: {len(train_dataset)}")
    if len(train_dataset) > 0:
        first_item = train_dataset[0]
        print(f"Type of first item: {type(first_item)}")
        print(f"Type of first item's x attribute: {type(first_item.x)}")
        print(f"Shape of first item's x attribute (if tensor): {first_item.x.shape if isinstance(first_item.x, torch.Tensor) else 'Not a tensor'}")
        print(f"Value of first item's x attribute: {first_item.x}")
