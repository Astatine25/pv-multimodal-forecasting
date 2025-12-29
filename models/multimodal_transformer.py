# models/multimodal_transformer.py

import tensorflow as tf
from tensorflow.keras import layers, models, Input

# -------------------------------
# Quantile loss
# -------------------------------
def quantile_loss(q):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    return loss


# -------------------------------
# Transformer encoder block
# -------------------------------
def transformer_block(x, d_model=64, num_heads=4, ff_dim=128):
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads
    )(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(d_model)(ff)
    x = layers.Add()([x, ff])
    return layers.LayerNormalization()(x)


# -------------------------------
# Multimodal Transformer Model
# -------------------------------
def build_model(history_steps, weather_dim, img_emb_dim=768, future_steps=6):
    """
    Inputs:
    - Temporal: (history_steps, 1 + weather_dim)
    - Image embedding: (img_emb_dim,)
    Outputs:
    - 3 quantiles × future_steps
    """

    # -------------------------------
    # Temporal branch
    # -------------------------------
    seq_in = Input(shape=(history_steps, 1 + weather_dim))
    x = layers.Dense(64)(seq_in)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)

    # -------------------------------
    # Image branch (ViT embeddings)
    # -------------------------------
    img_in = Input(shape=(img_emb_dim,))
    img_feat = layers.Dense(64, activation="relu")(img_in)

    # -------------------------------
    # Fusion
    # -------------------------------
    fusion = layers.Concatenate()([x, img_feat])
    fusion = layers.Dense(128, activation="relu")(fusion)
    fusion = layers.Dense(64, activation="relu")(fusion)

    # -------------------------------
    # Quantile outputs
    # -------------------------------
    out = layers.Dense(future_steps * 3)(fusion)
    out = layers.Reshape((3, future_steps))(out)

    model = models.Model([seq_in, img_in], out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=[
            quantile_loss(0.1),
            quantile_loss(0.5),
            quantile_loss(0.9),
        ]
    )

    return model
