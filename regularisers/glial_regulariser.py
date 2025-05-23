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


"""
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





