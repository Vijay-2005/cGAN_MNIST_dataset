import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Define Paths
HR_PATH = '/content/drive/MyDrive/DIV2K_HR/DIV2K_train_HR'  # Path to High-Resolution images
OUTPUT_DIR = './generated_images/'  # Output directory for generated images
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data Preprocessing
def load_hr_images(path, target_size=(256, 256)):
    images = []
    for img_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_name))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            images.append(img / 255.0)  # Normalize to [0, 1]
    return np.array(images, dtype=np.float32)

def generate_lr_images(hr_images, scale=4):
    """
    Generate LR images dynamically using TensorFlow operations.
    Args:
        hr_images: High-resolution images (tf.Tensor).
        scale: Downscaling factor (int).
    Returns:
        tf.Tensor: Low-resolution images.
    """
    # Ensure hr_images is a 4D tensor
    if len(hr_images.shape) != 4:
        hr_images = tf.expand_dims(hr_images, axis=0)

    # Downscale HR images
    lr_images = tf.image.resize(
        hr_images,
        [hr_images.shape[1] // scale, hr_images.shape[2] // scale],
        method=tf.image.ResizeMethod.BICUBIC
    )

    # Upscale back to original size
    lr_images = tf.image.resize(
        lr_images,
        [hr_images.shape[1], hr_images.shape[2]],
        method=tf.image.ResizeMethod.BICUBIC
    )
    return lr_images

hr_images = load_hr_images(HR_PATH)
hr_images = tf.convert_to_tensor(hr_images, dtype=tf.float32)  # Convert to TensorFlow tensor
lr_images = generate_lr_images(hr_images)

# Define Generator
def build_generator():
    inputs = layers.Input(shape=(256, 256, 3))
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.PReLU()(x)

    for _ in range(8):
        skip = x
        x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.PReLU()(x)
        x = layers.Add()([skip, x])

    x = layers.Conv2D(3, kernel_size=3, strides=1, padding='same')(x)
    outputs = layers.Activation('tanh')(x)
    return Model(inputs, outputs, name='Generator')

# Define Discriminator
def build_discriminator():
    inputs = layers.Input(shape=(256, 256, 3))
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU()(x)

    for filters in [128, 256, 512]:
        x = layers.Conv2D(filters, kernel_size=3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name='Discriminator')

# CGAN Class
class CGAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(CGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.bce = tf.keras.losses.BinaryCrossentropy()
        self.mse = tf.keras.losses.MeanSquaredError()

    def compile(self, g_optimizer, d_optimizer):
        super(CGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def train_step(self, data):
        lr_images, hr_images = data  # Unpack LR and HR images from the dataset

        # Train Discriminator
        with tf.GradientTape() as d_tape:
            fake_hr = self.generator(lr_images, training=True)
            real_pred = self.discriminator(hr_images, training=True)
            fake_pred = self.discriminator(fake_hr, training=True)
            d_loss_real = self.bce(tf.ones_like(real_pred), real_pred)
            d_loss_fake = self.bce(tf.zeros_like(fake_pred), fake_pred)
            d_loss = d_loss_real + d_loss_fake

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # Train Generator
        with tf.GradientTape() as g_tape:
            fake_hr = self.generator(lr_images, training=True)
            fake_pred = self.discriminator(fake_hr, training=True)
            g_loss_adv = self.bce(tf.ones_like(fake_pred), fake_pred)
            g_loss_content = self.mse(hr_images, fake_hr)
            g_loss = g_loss_adv + g_loss_content

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

# Callback for Saving Generated Images
class SaveGeneratedImages(tf.keras.callbacks.Callback):
    def __init__(self, generator, output_dir, test_lr_images):
        self.generator = generator
        self.output_dir = output_dir
        self.test_lr_images = test_lr_images

    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.generator.predict(self.test_lr_images)
        for i, img in enumerate(generated_images):
            plt.imsave(f"{self.output_dir}/epoch_{epoch+1}_img_{i+1}.png", (img * 255).astype(np.uint8))

# Instantiate Models
generator = build_generator()
discriminator = build_discriminator()
cgan = CGAN(generator, discriminator)
cgan.compile(
    g_optimizer=tf.keras.optimizers.Adam(1e-4),
    d_optimizer=tf.keras.optimizers.Adam(1e-4)
)

# Prepare Test Data for Image Saving
test_lr_images = generate_lr_images(tf.convert_to_tensor(hr_images[:5]))  # Use first 5 HR images for visualization
save_callback = SaveGeneratedImages(generator, OUTPUT_DIR, test_lr_images)

# Combine LR and HR images into a dataset
dataset = tf.data.Dataset.from_tensor_slices((lr_images, hr_images)).batch(16)

# Train the CGAN
cgan.fit(dataset, epochs=100, callbacks=[save_callback])
