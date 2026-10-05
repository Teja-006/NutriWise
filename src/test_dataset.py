from dataset import get_dataloaders


train_loader, val_loader, classes = get_dataloaders(
    data_dir="data/Indian Food",
    batch_size=32
)


images, labels = next(iter(train_loader))


print("\nBatch image shape:")
print(images.shape)

print("\nBatch label shape:")
print(labels.shape)

print("\nNumber of classes:")
print(len(classes))

print("\nFirst 10 classes:")
print(classes[:10])