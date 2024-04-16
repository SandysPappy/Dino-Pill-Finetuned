from dataset_loaders import get_epill_dataloader

loader = get_epill_dataloader(fold='refs', batch_size=15, use_epill_transforms=True)

print(len(loader))
# print(len(loader[0]))

for batch in loader:
    #print(len(batch['image']), len(batch['label']), len(batch['is_front']), len(batch['is_ref']))
    # print(batch[0]["image"].shape)
    print('batch["image"].shape:    ', batch['image'].shape)
    print('batch["label"].shape:    ', batch['label'].shape)
    print('batch["is_front"].shape: ', batch['is_front'].shape)
    print('batch["is_ref"].shape:   ', batch['is_ref'].shape)
    