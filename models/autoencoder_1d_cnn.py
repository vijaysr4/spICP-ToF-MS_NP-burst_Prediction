from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

def build_autoencoder(
    window_size: int,
    n_channels: int,
    latent_dim: int = 16
) -> models.Model:
    """
    Construct and compile a 1D convolutional autoencoder for multivariate time-series windows.

    Args:
        window_size: Number of time bins in each input window.
        n_channels: Number of feature channels (e.g., isotope columns).
        latent_dim: Size of the bottleneck layer (controls compression).

    Returns:
        A compiled Keras Model that maps inputs (window_size, n_channels)
        to reconstructed outputs of the same shape.
    """
    # Input layer
    inputs = layers.Input(shape=(window_size, n_channels))

    # Encoder: two Conv1D + MaxPooling blocks
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Conv1D(16, kernel_size=3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling1D(pool_size=2, padding='same')(x)

    # Decoder: mirror of Encoder with UpSampling
    x = layers.Conv1D(16, kernel_size=3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(size=2)(x)

    # Output layer: reconstruct original channels
    outputs = layers.Conv1D(n_channels, kernel_size=3, activation='linear', padding='same')(x)

    # Build and compile model
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='mse')
    return model
