import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import custom_object_scope
from regularisers import Glial_1D_projective, Glial_1D_receptive, Glial_1D_p_r
from mnist1d.data import make_dataset, get_dataset_args
from sklearn.model_selection import train_test_split

# Seed for reproducibility
SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
MODEL_DIR = "/Users/OliverBenton/Repositories/1DMNIST/Models"
OUTPUT_PATH = "/Users/OliverBenton/Repositories/1DMNIST/Shared_Successful_Activations"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load data
defaults = get_dataset_args()
data = make_dataset(defaults)
train_images, test_images, train_labels, test_labels = train_test_split(
    data["x"], data["y"], test_size=0.2, random_state=SEED
)
test_images = test_images / np.max(test_images, axis=-1, keepdims=True)

# Load models
model_paths = {
    "L2": os.path.join(MODEL_DIR, "L2_activations_40x40.keras"),
    "Glial": os.path.join(MODEL_DIR, "Glial_activations_40x40.keras"),
    "NoReg": os.path.join(MODEL_DIR, "No_Reg_activations_40x40.keras"),
}

models = {}
with custom_object_scope({
    'Glial_1D_receptive': Glial_1D_receptive,
    'Glial_1D_projective': Glial_1D_projective,
    'Glial_1D_p_r': Glial_1D_p_r
}):
    for name, path in model_paths.items():
        models[name] = tf.keras.models.load_model(path)

# Get predictions
predictions = {
    name: np.argmax(model.predict(test_images, verbose=0), axis=1)
    for name, model in models.items()
}

# Find shared correct indices by digit
shared_indices_by_digit = {}
for digit in range(10):
    shared_indices = np.where(test_labels == digit)[0]
    for name in models:
        shared_indices = shared_indices[
            predictions[name][shared_indices] == digit
        ]
    if len(shared_indices) > 0:
        shared_indices_by_digit[digit] = shared_indices
    else:
        # fallback: pick any correct prediction if no shared ones
        print(f"No shared correct samples for digit {digit}. Falling back to L2 correct prediction.")
        fallback_indices = np.where((test_labels == digit) & (predictions['L2'] == digit))[0]
        if len(fallback_indices) > 0:
            shared_indices_by_digit[digit] = fallback_indices[:1]

# Print shared info
print("Shared indices by digit:")
for digit, indices in shared_indices_by_digit.items():
    print(f"{digit}: {len(indices)} sample(s) used")

# Extract activations layer by layer
def get_activations(model, input_data):
    activations = []
    for layer in model.layers:
        input_data = layer(input_data)
        activations.append(input_data.numpy())
    return activations

# Main plotting function
def plot_shared_activations(models, X_data, shared_indices_by_digit):
    model_order = ['NoReg', 'L2', 'Glial']
    layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
    num_digits = len(shared_indices_by_digit)
    fig, axs = plt.subplots(num_digits * 3, 4, figsize=(20, 3 * num_digits * 3))

    def top_k_indices(act, k=5):
        return set(np.argsort(act.flatten())[-k:])

    def get_sparse_colored_bar_data(acts_by_model, size):
        top_indices = {model: top_k_indices(act, k=5) for model, act in acts_by_model.items()}
        shared = set.intersection(*top_indices.values())

        model_data = {}
        for model, act in acts_by_model.items():
            flat = act.flatten()
            values = np.zeros(size)
            colors = ['lightgray'] * size
            for i in top_indices[model]:
                values[i] = flat[i]
                colors[i] = 'red' if i in shared else 'steelblue'
            model_data[model] = (values, colors)
        return model_data

    for digit_idx, digit in enumerate(sorted(shared_indices_by_digit.keys())):
        sample_idx = shared_indices_by_digit[digit][0]
        sample = X_data[sample_idx:sample_idx+1]

        if sample.ndim == 2:
            sample = sample.reshape(1, -1)

        activations_by_model = {}
        for model_name in model_order:
            model = models[model_name]
            activations_by_model[model_name] = get_activations(model, sample)

        shared_layer_data = {}
        for layer_idx, size in zip([0, 1, 2], [40, 40, 10]):
            acts = {model: activations_by_model[model][layer_idx] for model in model_order}
            shared_layer_data[layer_idx] = get_sparse_colored_bar_data(acts, size)

        for row_idx, model_name in enumerate(model_order):
            base_row = digit_idx * 3 + row_idx
            acts = activations_by_model[model_name]

            # Input (only once per digit)
            if row_idx == 0:
                axs[base_row, 0].plot(sample[0])
                axs[base_row, 0].set_ylim(-1, 1)
            else:
                axs[base_row, 0].axis('off')

            # Hidden 1
            vals, colors = shared_layer_data[0][model_name]
            axs[base_row, 1].bar(range(40), vals, color=colors)

            # Hidden 2
            vals, colors = shared_layer_data[1][model_name]
            axs[base_row, 2].bar(range(40), vals, color=colors)

            # Output
            vals, colors = shared_layer_data[2][model_name]
            axs[base_row, 3].bar(range(10), vals, color=colors)

            # Titles
            if base_row == 0:
                for col_idx in range(4):
                    axs[base_row, col_idx].set_title(layer_names[col_idx], fontsize=14)

            # Label digit once
            if row_idx == 0:
                axs[base_row, 0].set_ylabel(f"{digit}", fontsize=12, rotation=0, labelpad=30)

            # Label model
            axs[base_row, 0].annotate(model_name, xy=(-0.15, 0.5), xycoords='axes fraction',
                                      fontsize=11, ha='center', va='center', rotation=90)

    plt.tight_layout()
    plt.suptitle("Top 5 Activations Highlighted Across Models", fontsize=24, y=1.02)
    plt.subplots_adjust(top=0.96)
    plt.savefig(os.path.join(OUTPUT_PATH, f"Top5_Highlighted_Shared_Activations_{SEED}.png"))
    plt.close()

# Run the plotting
plot_shared_activations(models, test_images, shared_indices_by_digit)
