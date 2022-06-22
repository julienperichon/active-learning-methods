#%%

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

#%%

(train_ds, val_ds, test_ds, unlabelled_ds), metadata = tfds.load(
    "tf_flowers",
    split=["train[:1000]", "train[1000:1500]", "train[1500:2500]", "train[2500:]"],
    with_info=True,
    as_supervised=True,
)
n_classes = metadata.features["label"].num_classes

#%%

resizing_layer = tf.keras.layers.Resizing(299, 299)


def preprocess_images(dataset):
    ds = dataset.cache()
    ds = ds.map(lambda x, y: (resizing_layer(x), y))
    ds = ds.map(lambda x, y: (x / 255.0, y))
    return ds


def batch_dataset(dataset):
    ds = dataset.batch(100)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


#%%

preprocessed_train_ds = preprocess_images(train_ds)
train_ds = batch_dataset(preprocessed_train_ds)
val_ds = batch_dataset(preprocess_images(val_ds))
test_ds = batch_dataset(preprocess_images(test_ds))
unlabelled_ds = preprocess_images(unlabelled_ds)

#%%

model = tf.keras.Sequential(
    [
        hub.KerasLayer(
            "https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5",
            trainable=False,
        ),
        tf.keras.layers.Dense(n_classes),
    ]
)
model.build([None, 299, 299, 3])  # Batch input shape.

#%%

model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

#%%

model.fit(train_ds, validation_data=val_ds, epochs=3)

#%%

model.trainable = True
model.fit(train_ds, validation_data=val_ds, epochs=2)

#%%

pred_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

#%%

_, accuracy = model.evaluate(test_ds)
print(f"Test accuracy after first training: {accuracy:.2%}")

#%% Least confidence selection

predictions = [
    pred_model.predict(batch_images)
    for batch_images, batch_labels in batch_dataset(unlabelled_ds)
]
preds = np.concatenate(predictions)
best_confidence = preds.max(axis=1)

#%%

budget = 500
least_confidence_indexes = np.argsort(best_confidence)[:budget]


#%%

images, labels = tuple(zip(*unlabelled_ds))
chosen_images = np.array(images)[least_confidence_indexes]
chosen_labels = np.array(labels)[least_confidence_indexes]
least_confidence_dataset = tf.data.Dataset.from_tensor_slices(
    (chosen_images, chosen_labels)
)

#%%

_, accuracy = model.evaluate(batch_dataset(least_confidence_dataset))
print(f"Accuracy on data with least confidence: {accuracy:.2%}")

#%%

new_train_ds = preprocessed_train_ds.concatenate(least_confidence_dataset)
new_train_ds = batch_dataset(new_train_ds)
model.fit(new_train_ds, validation_data=val_ds, epochs=3)

#%%
_, accuracy = model.evaluate(test_ds)
print(f"Test accuracy after second training: {accuracy:.2%}")

#%%
_, accuracy = model.evaluate(batch_dataset(least_confidence_dataset))
print(f"Accuracy on data with least confidence: {accuracy:.2%}")


#%%

predictions = [
    pred_model.predict(batch_images)
    for batch_images, batch_labels in batch_dataset(unlabelled_ds)
]
preds = np.concatenate(predictions)
best_confidence = preds.max(axis=1)

#%%

budget = 500
least_confidence_indexes = np.argsort(best_confidence)[:budget]
