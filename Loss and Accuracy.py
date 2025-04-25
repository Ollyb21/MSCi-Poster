import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from regularisers.glial_regulariser import Glial_1D_receptive
from regularisers.glial_regulariser import Glial_1D_projective
from regularisers.glial_regulariser import Glial_1D_p_r


import mnist1d
from mnist1d.data import make_dataset, get_dataset_args
from sklearn.model_selection import train_test_split

# Set a fixed random seed for reproducibility
SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)

r_strength = 0.000
p_strength = 0.000
l2_strength = 0.000

"""Parameters for training model"""
#Glial_1D_r = Glial_1D_receptive(strength=0, smoothness_strength=0.01011881251140045)
#Glial_1D_p = Glial_1D_projective(strength=0, smoothness_strength=0.034930316059835674)
Glial_1D_pr = Glial_1D_p_r(smoothness_strength_p=p_strength, smoothness_strength_r=r_strength)


defaults = get_dataset_args()
data = make_dataset(defaults)

# Deep Layers Params
reg_1 = regularizers.l2(l2_strength)
#reg_1 = Glial_1D_pr
act_1 = "relu"
act_2 = "relu"

epochs = 250

# Compile the model
opt = keras.optimizers.Adam(learning_rate=0.00630755146333121)
lose = "sparse_categorical_crossentropy"
met = ["accuracy"]

layer1 = 40
layer2 = 40

"""Preprocessing of data"""
train_images, test_images, train_labels, test_labels = train_test_split(
    data["x"], data["y"], test_size=0.2, random_state=SEED
)
print(f"Train images shape: {train_images.shape}")
print(f"Test images shape: {test_images.shape}")

"""Model Architecture"""
model = keras.Sequential([
    layers.Dense(layer1, activation=act_1, kernel_regularizer=reg_1),
    layers.Dense(layer2, activation=act_2, kernel_regularizer=reg_1),
    layers.Dense(10, activation="softmax")
    ])

"""Compile and train the model"""
model.compile(optimizer=opt, loss=lose, metrics=met)

history = model.fit(train_images, train_labels, epochs=epochs, batch_size=43, validation_split=0.2)

"""Evaluate the model"""
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc}")
print(f"Test Loss: {test_loss}")

"""Save the model"""
model.save('/Users/OliverBenton/Repositories/1DMNIST/Models/No_Reg_activations_40x40.keras')

"""Create subplots for loss and accuracy"""
plt.figure(figsize=(12, 5))

# Accuracy subplot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title(f"ACCURACY: r = {r_strength} p = {p_strength} (40/{layer1}/{layer2}/10)")
plt.figtext(0.15, 0.12, f"Accuracy (Test Acc: {test_acc:.4f})", fontsize=10, ha='left', va='bottom')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend()

# Loss subplot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title(f"LOSS: r = {r_strength} p = {p_strength} (40/{layer1}/{layer2}/10)")
plt.figtext(0.85, 0.12, f"Loss (Test Loss: {test_loss:.4f})", fontsize=10, ha='right', va='bottom')
plt.ylim(0, 5)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()


plt.tight_layout()
plt.savefig(f"/Users/OliverBenton/Repositories/1DMNIST/Poster data/Trial[{SEED}]{epochs} epochs with r = {r_strength} p = {p_strength} (40:{layer1}:{layer2}:10) l2 = {l2_strength}.png")
plt.show()