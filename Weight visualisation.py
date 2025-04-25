import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from keras.utils import custom_object_scope
from regularisers import Glial_1D_projective, Glial_1D_receptive, Glial_1D_p_r

# Define model paths
MODEL_DIR = "/Users/OliverBenton/Repositories/1DMNIST/Models"
model_paths = {
    "L2": os.path.join(MODEL_DIR, "L2_activations_40x40.keras"),
    "Glial": os.path.join(MODEL_DIR, "Glial_activations_40x40.keras"),
    "No": os.path.join(MODEL_DIR, "No_Reg_activations_40x40.keras"),
}

# Set up seaborn style
sns.set(style="whitegrid")

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

with custom_object_scope({
    'Glial_1D_receptive': Glial_1D_receptive,
    'Glial_1D_projective': Glial_1D_projective,
    'Glial_1D_p_r': Glial_1D_p_r
}):
    for ax, (name, path) in zip(axes, model_paths.items()):
        model = tf.keras.models.load_model(path)
        second_layer_weights, _ = model.layers[1].get_weights()

        sns.heatmap(
            second_layer_weights,
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            annot=False,
            square=True,
            ax=ax,
            cbar=False
        )
        ax.set_title(f"{name} Regularization")
        ax.set_xlabel("Neuron in 2nd Hidden Layer")
        ax.set_ylabel("Neuron in 1st Hidden Layer")

# Set one colorbar on the right
fig.colorbar(
    plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-1, vmax=1)),
    ax=axes,
    location="right",
    shrink=0.6,
    label="Weight Strength"
)

fig.suptitle("Second Layer Weights Across Regularizations", fontsize=16)
#plt.savefig("/Users/OliverBenton/Repositories/1DMNIST/Poster Data/Second_Layer_Weights.png", dpi=300)
plt.show()

