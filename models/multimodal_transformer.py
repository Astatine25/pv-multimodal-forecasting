# models/multimodal_transformer.py
import tensorflow as tf
from tensorflow.keras import layers, models, Input

# -----------------------------
# GLOBAL PARAMETERS
# -----------------------------
HISTORY_STEPS = 12
FUTURE_STEPS = 6
IMG_EMB_DIM = 768  # ViT default embedding size
D_MODEL = 64

# -----------------------------
# Quantile loss function
# -----------------------------
def quantile_loss(q):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    return loss

# -----------------------------
# Transformer Encoder Block
# -----------------------------
def transformer_encoder(x, d_model, num_heads, ff_dim, dropout=0.1):
    # Multi-head self-attention
    attn_out = layers.MultiHeadAttention(
        key_dim=d_model // num_heads,
        num_heads=num_heads,
        dropout=dropout
    )(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Feed-forward network
    ff_out = layers.Dense(ff_dim, activation="relu")(x)
    ff_out = layers.Dense(d_model)(ff_out)
    x = layers.Add()([x, ff_out])
    return layers.LayerNormalization(epsilon=1e-6)(x)

# -----------------------------
# Build Multimodal Transformer
# -----------------------------
def build_model(history_steps=HISTORY_STEPS, num_weather_features=3, future_steps=FUTURE_STEPS, img_emb_dim=IMG_EMB_DIM):
    # -------- Temporal Input: Power + Weather --------
    seq_in = Input(shape=(history_steps, 1 + num_weather_features))
    x = layers.Dense(D_MODEL)(seq_in)
    x = transformer_encoder(x, D_MODEL, num_heads=4, ff_dim=128)
    x = layers.GlobalAveragePooling1D()(x)

    # -------- Image Embedding Input (ViT) --------
    img_in = Input(shape=(img_emb_dim,))
    img_feat = layers.Dense(D_MODEL, activation="relu")(img_in)

    # -------- Fusion --------
    fusion = layers.Concatenate()([x, img_feat])
    fusion = layers.Dense(128, activation="relu")(fusion)
    fusion = layers.Dense(64, activation="relu")(fusion)

    # -------- Output: 3 Quantiles --------
    out = layers.Dense(future_steps * 3)(fusion)
    out = layers.Reshape((3, future_steps))(out)

    model = models.Model([seq_in, img_in], out)

    # Compile with median quantile loss (0.5) as main optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=quantile_loss(0.5)
    )
    return model
