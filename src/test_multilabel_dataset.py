from dataset import build_datasets


train_ds, val_ds, test_ds, class_names = build_datasets(
    data_dir="data"
)


print("\n==============================")
print("MULTI-LABEL DATASET TEST")
print("==============================")


print("Training images:", len(train_ds))

print("Validation images:", len(val_ds))

print("Testing images:", len(test_ds))

print("Classes:", len(class_names))


image, labels = train_ds[0]


print("\nImage shape:")

print(image.shape)


print("\nLabel shape:")

print(labels.shape)


print("\nLabels:")

print(labels)


print("\nPresent classes:")


for i, value in enumerate(labels):

    if value == 1:

        print(class_names[i])