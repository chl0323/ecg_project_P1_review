# new_transformer_feature_extractor.py
# Transformer feature extractor implementation for ECG deep learning project.
# All code, comments, docstrings, and variable names are fully in English for academic submission.

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class PositionalEncoding:
    def __init__(self, position, d_model):
        self.position = position
        self.d_model = d_model
        
    def __call__(self):
        # Create trainable positional encoding
        pos_encoding = layers.Embedding(self.position, self.d_model)(tf.range(self.position, dtype=tf.int32))
        pos_encoding = tf.expand_dims(pos_encoding, 0)  # Add batch dimension
        return pos_encoding

class TransformerFeatureExtractor:
    def __init__(self, input_dim, sequence_length, num_heads=4, d_model=128, rate=0.1):
        """
        Transformer-based feature extractor for sequential ECG data.
        Args:
            input_dim (int): Number of input features per time step.
            sequence_length (int): Length of the input sequence.
            num_heads (int): Number of attention heads.
            d_model (int): Dimension of the model.
            rate (float): Dropout rate.
        """
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.num_heads = num_heads
        self.d_model = d_model
        self.rate = rate
        self.attention_layers = []  # Store attention layer objects
        # Initialize trainable positional embedding
        self.position_embedding = layers.Embedding(
            input_dim=sequence_length,
            output_dim=input_dim,
            name='position_embedding'
        )

    def transformer_encoder_layer(self, x, num_heads, d_model, rate):
        """
        Single transformer encoder layer with multi-head attention and feed-forward network.
        """
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.input_dim // num_heads
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attention_output)
        ffn_output = self.point_wise_feed_forward_network(self.input_dim, d_model)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)
        return x

    def point_wise_feed_forward_network(self, d_model, dff):
        """
        Point-wise feed-forward network for transformer encoder.
        """
        return tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

    def build(self):
        """
        Build the transformer feature extractor model.
        Returns:
            tf.keras.Model: The feature extractor model.
        """
        inputs = tf.keras.layers.Input(shape=(self.sequence_length, self.input_dim))
        # Create position indices
        position_indices = tf.keras.layers.Lambda(
            lambda x: tf.range(start=0, limit=self.sequence_length, delta=1),
            output_shape=(self.sequence_length,)
        )(inputs)
        position_indices = tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x[1], axis=0),
            output_shape=(1, self.sequence_length)
        )([inputs, position_indices])
        position_indices = tf.keras.layers.Lambda(
            lambda x: tf.tile(x[1], [tf.shape(x[0])[0], 1]),
            output_shape=(None, self.sequence_length)
        )([inputs, position_indices])
        # Add positional encoding
        pos_encoding = self.position_embedding(position_indices)
        x = tf.keras.layers.Add()([inputs, pos_encoding])
        # Transformer encoder layers
        self.attention_layers = []
        for i in range(2):  # Use 2 transformer encoder layers
            attn_layer = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.input_dim // self.num_heads,
                name=f'multi_head_attention_{i}'
            )
            attn = attn_layer(x, x, return_attention_scores=False)
            self.attention_layers.append(attn_layer)
            x = tf.keras.layers.LayerNormalization(name=f'layer_norm_1_{i}')(tf.keras.layers.Add()([x, attn]))
            ffn = tf.keras.layers.Dense(self.d_model, activation='relu', name=f'dense_1_{i}')(x)
            ffn = tf.keras.layers.Dense(self.input_dim, name=f'dense_2_{i}')(ffn)
            x = tf.keras.layers.LayerNormalization(name=f'layer_norm_2_{i}')(tf.keras.layers.Add()([x, ffn]))
            x = tf.keras.layers.Dropout(self.rate)(x)
        x = tf.keras.layers.Reshape((self.sequence_length, self.input_dim))(x)
        # Dual-channel aggregation: mean and max pooling
        mean_pool = tf.keras.layers.GlobalAveragePooling1D(name='mean_pool')(x)
        max_pool = tf.keras.layers.GlobalMaxPooling1D(name='max_pool')(x)
        combined = tf.keras.layers.Concatenate(name='concat')([mean_pool, max_pool])
        embedding = tf.keras.layers.Dense(self.input_dim, activation='tanh', name='final_dense')(combined)
        return tf.keras.Model(inputs, embedding)

    def get_config(self):
        """
        Return configuration for serialization.
        """
        return {
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'rate': self.rate
        }

    @classmethod
    def from_config(cls, config):
        """
        Create an instance from configuration.
        """
        return cls(**config)

    def load_weights(self, weights_path):
        """
        Load model weights, handling potential weight mismatches.
        """
        try:
            self.model.load_weights(weights_path)
            print(f"Successfully loaded weights from file: {weights_path}")
        except Exception as e:
            print(f"Failed to load weights directly: {str(e)}")
            print("Attempting to load weights in compatible mode...")
            try:
                saved_model = tf.keras.models.load_model(weights_path, compile=False)
                saved_weights_dict = {layer.name: layer.get_weights() for layer in saved_model.layers}
            except Exception as e:
                print(f"Failed to load full model: {str(e)}")
                print("Attempting to load weights from file...")
                if weights_path.endswith('.weights.h5'):
                    try:
                        self.model.load_weights(weights_path)
                        print(f"Successfully loaded weights from file: {weights_path}")
                        return
                    except Exception as e:
                        print(f"Failed to load weights from file: {str(e)}")
                        raise
            current_layers = {layer.name: layer for layer in self.model.layers}
            loaded_count = 0
            for layer_name, weights in saved_weights_dict.items():
                if layer_name in current_layers:
                    try:
                        current_layers[layer_name].set_weights(weights)
                        loaded_count += 1
                        print(f"Successfully loaded weights for layer {layer_name}")
                    except Exception as e:
                        print(f"Failed to load weights for layer {layer_name}: {str(e)}")
            print(f"Weights loaded, successfully loaded weights for {loaded_count} layers")
            if loaded_count == 0:
                raise ValueError("No weights were successfully loaded")

    def get_attention_scores(self, x_input, layer_idx=0):
        """
        Get attention scores for a specified layer.
        Args:
            x_input (np.ndarray or tf.Tensor): Input data of shape (batch, seq_len, input_dim).
            layer_idx (int): Index of the attention layer (0-based).
        Returns:
            np.ndarray: Attention scores of shape (batch, num_heads, seq_len, seq_len).
        """
        if not isinstance(x_input, tf.Tensor):
            x_input = tf.convert_to_tensor(x_input, dtype=tf.float32)
        if len(x_input.shape) != 3:
            raise ValueError(f"Input data should be 3D (batch_size, sequence_length, features), but got {x_input.shape}")
        if x_input.shape[1] != self.sequence_length:
            raise ValueError(f"Sequence length should be {self.sequence_length}, but got {x_input.shape[1]}")
        if x_input.shape[2] != self.input_dim:
            raise ValueError(f"Feature dimension should be {self.input_dim}, but got {x_input.shape[2]}")
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        positions = tf.expand_dims(positions, axis=0)
        positions = tf.tile(positions, [tf.shape(x_input)[0], 1])
        pos_encoding = self.position_embedding(positions)
        x_input = x_input + pos_encoding
        attn_layer = self.attention_layers[layer_idx]
        _, attn_scores = attn_layer(x_input, x_input, return_attention_scores=True)
        return attn_scores.numpy()