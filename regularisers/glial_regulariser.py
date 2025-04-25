import tensorflow as tf
from tensorflow.keras import regularizers

class Glial_1D_projective(regularizers.Regularizer):
    def __init__(self, strength, smoothness_strength):
        self.strength = strength  # L2 regularization strength
        self.smoothness_strength = smoothness_strength  # Smoothness penalty strength

    def __call__(self, weights):
        #reg_loss = self.strength * tf.reduce_sum(tf.square(weights))  # L2 penalty
        
        diff_left = weights[:, 1:-1] - weights[:, :-2]  # W[:, i] - W[:, i-1]
        diff_right = weights[:, 1:-1] - weights[:, 2:]   # W[:, i] - W[:, i+1]
        smoothness_loss = self.smoothness_strength * (tf.reduce_sum(tf.square(diff_left)) + tf.reduce_sum(tf.square(diff_right)))
        
        return smoothness_loss
#reg_loss +
    def get_config(self):  # Required for saving/loading models
        return {"strength": self.strength, "smoothness_strength": self.smoothness_strength}


class Glial_1D_receptive(regularizers.Regularizer):
    def __init__(self, strength, smoothness_strength):
        self.strength = strength  # L2 regularization strength
        self.smoothness_strength = smoothness_strength  # Smoothness penalty strength

    def __call__(self, weights):
        #reg_loss = self.strength * tf.reduce_sum(tf.square(weights))  # L2 penalty
        
        # Smoothness term: penalize large differences between neighboring weights
        diff_up = weights[1:-1, :] - weights[:-2, :]  # W[i, :] - W[i-1, :]
        diff_down = weights[1:-1, :] - weights[2:, :] # W[i, :] - W[i+1, :]
        smoothness_loss = self.smoothness_strength * (tf.reduce_sum(tf.square(diff_up)) + tf.reduce_sum(tf.square(diff_down)))   
        
        return smoothness_loss
#reg_loss +
    def get_config(self):  # Required for saving/loading models
        return {"strength": self.strength, "smoothness_strength": self.smoothness_strength}




class Glial_1D_p_r(regularizers.Regularizer):
    def __init__(self, smoothness_strength_p, smoothness_strength_r):
        self.smoothness_strength_p = smoothness_strength_p  # Smoothness penalty strength projective
        self.smoothness_strength_r = smoothness_strength_r  # Smoothness penalty strength receptive

    def __call__(self, weights):
        
        diff_up_r = weights[1:-1, :] - weights[:-2, :]  # W[i, :] - W[i-1, :]
        diff_down_r = weights[1:-1, :] - weights[2:, :] # W[i, :] - W[i+1, :]
        smoothness_loss_r = self.smoothness_strength_r * (tf.reduce_sum(tf.square(diff_up_r)) + tf.reduce_sum(tf.square(diff_down_r)))   
        
        diff_left_p = weights[:, 1:-1] - weights[:, :-2]  # W[:, i] - W[:, i-1]
        diff_right_p = weights[:, 1:-1] - weights[:, 2:]   # W[:, i] - W[:, i+1]
        smoothness_loss_p = self.smoothness_strength_p * (tf.reduce_sum(tf.square(diff_left_p)) + tf.reduce_sum(tf.square(diff_right_p)))
        
        return smoothness_loss_p + smoothness_loss_r

    def get_config(self):  # Required for saving/loading models
        return {"smoothness_strength_p": self.smoothness_strength_p, "smoothness_strength_r": self.smoothness_strength_r}



class Glial_1D_p_r_window_size(regularizers.Regularizer):
    def __init__(self, smoothness_strength_p, smoothness_strength_r, window_size_p=1, window_size_r=1):
        self.smoothness_strength_p = smoothness_strength_p  # Smoothness penalty strength projective
        self.smoothness_strength_r = smoothness_strength_r  # Smoothness penalty strength receptive
        self.window_size_p = window_size_p  # Window size for projective smoothness
        self.window_size_r = window_size_r  # Window size for receptive smoothness

    def __call__(self, weights):
        smoothness_loss_p = 0.0
        smoothness_loss_r = 0.0
        
        # Compute smoothness loss for projective connections
        for i in range(1, self.window_size_p + 1):
            diff_p = weights[:, i:] - weights[:, :-i]  # Compute Wi+k - Wi along neuron connections
            smoothness_loss_p += self.smoothness_strength_p * tf.reduce_sum(tf.square(diff_p)) / i
        
        # Compute smoothness loss for receptive connections
        for i in range(1, self.window_size_r + 1):
            diff_r = weights[i:, :] - weights[:-i, :]  # Compute Wi+k - Wi along neuron connections
            smoothness_loss_r += self.smoothness_strength_r * tf.reduce_sum(tf.square(diff_r)) / i
        
        return smoothness_loss_p + smoothness_loss_r

    def get_config(self):  # Required for saving/loading models
        return {
            "smoothness_strength_p": self.smoothness_strength_p,
            "smoothness_strength_r": self.smoothness_strength_r,
            "window_size_p": self.window_size_p,
            "window_size_r": self.window_size_r
        }








"""
class SmoothnessScheduler(tf.keras.callbacks.Callback):
    def __init__(self, reg, increase_factor=1.05, decrease_factor=0.95, min_strength=1e-5, max_strength=0.1):
        self.reg = reg
        self.increase_factor = increase_factor  # How much to increase smoothness if validation loss improves
        self.decrease_factor = decrease_factor  # How much to decrease smoothness if validation loss worsens
        self.min_strength = min_strength  # Prevents smoothness from going too low
        self.max_strength = max_strength  # Prevents smoothness from going too high
        self.prev_val_loss = float('inf')  # Track validation loss

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")  # Get current validation loss

        if val_loss is not None:
            # If validation loss decreased → increase smoothness strength
            if val_loss < self.prev_val_loss:
                if hasattr(self.reg, 'smoothness_strength_p'):
                    self.reg.smoothness_strength_p = min(self.reg.smoothness_strength_p * self.increase_factor, self.max_strength)
                if hasattr(self.reg, 'smoothness_strength_r'):
                    self.reg.smoothness_strength_r = min(self.reg.smoothness_strength_r * self.increase_factor, self.max_strength)
            else:
                # If validation loss increased → decrease smoothness strength
                if hasattr(self.reg, 'smoothness_strength_p'):
                    self.reg.smoothness_strength_p = max(self.reg.smoothness_strength_p * self.decrease_factor, self.min_strength)
                if hasattr(self.reg, 'smoothness_strength_r'):
                    self.reg.smoothness_strength_r = max(self.reg.smoothness_strength_r * self.decrease_factor, self.min_strength)

            self.prev_val_loss = val_loss  # Update validation loss tracker

        print(f"Epoch {epoch+1}: Updated smoothness strengths -> p={self.reg.smoothness_strength_p:.6f}, r={self.reg.smoothness_strength_r:.6f}")
"""
