import os
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

# Hyperparameters
latent_dim = 32          # Size of the noise vector
data_dim = 10            # Number of features in your data
batch_size = 64
num_epochs = 5000
lr = 0.0002

# Self - Data Location
load_dotenv()
file_path = os.getenv('CSV_FILE_PATH')
gan_path = os.getenv('GAN_FILE_PATH')
doc_path = os.getenv('DOCS_FILE_PATH')

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, data_dim),
        )

    def forward(self, z):
        return self.model(z)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, data_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        return self.model(data)

def gan_model(data):
    # Data
    df = pd.read_csv(data)

    # Drop 'time' column
    df = df.drop(columns=['time'])

    # Handle NaN values (choose one strategy)
    df_filled = df.fillna(0) 

    # Normalize between -1 and 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df_filled)

    # Convert to Torch Tensor
    real_data = torch.tensor(scaled_data, dtype=torch.float32)

    # Set data_dim
    data_dim = real_data.shape[1]

    print(f"Data Dimension: {data_dim}, Shape: {real_data.shape}")

    # 🚀 Reinitialize the generator and discriminator with updated data_dim
    generator = Generator(latent_dim, data_dim)
    discriminator = Discriminator(data_dim)

    # ✔️ Optimizers again with new models
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    # Training Loop
    for epoch in range(num_epochs):
        # === Train Discriminator ===
        idx = np.random.randint(0, real_data.shape[0], batch_size)
        real_samples = real_data[idx]

        z = torch.randn(batch_size, latent_dim)
        fake_samples = generator(z)

        real_labels = torch.ones((batch_size, 1))
        fake_labels = torch.zeros((batch_size, 1))

        # Discriminator loss on real and fake
        outputs_real = discriminator(real_samples)
        loss_real = criterion(outputs_real, real_labels)

        outputs_fake = discriminator(fake_samples.detach())
        loss_fake = criterion(outputs_fake, fake_labels)

        loss_D = loss_real + loss_fake

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # === Train Generator ===
        z = torch.randn(batch_size, latent_dim)
        generated_samples = generator(z)

        outputs = discriminator(generated_samples)
        loss_G = criterion(outputs, real_labels)  # Fool discriminator into believing it's real

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # Logging
        if epoch % 500 == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch}/{num_epochs}] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

    # Generate Synthetic Data
    generator.eval()
    with torch.no_grad():
        z = torch.randn(10, latent_dim)
        synthetic_data = generator(z)
        print("\nSample Synthetic Data:\n", synthetic_data)

    real_values = real_data[:, 0].numpy()  # Example feature
    fake_values = synthetic_data[:, 0].numpy()

    # Convert to DataFrame and save
    synthetic_df = pd.DataFrame(synthetic_data.numpy(), columns=df.columns)

    # Save synthetic data to CSV
    synthetic_csv_path = gan_path+"/32-sorted.csv"
    synthetic_df.to_csv(synthetic_csv_path, index=False)

    print(f"Synthetic data CSV created: {data}")

    synthetic_csv_path

    sns.kdeplot(real_values, label="Real")
    sns.kdeplot(fake_values, label="Fake")
    plt.legend()
    plt.title(data)

    plt.savefig(os.path.join(doc_path+"/graphs", "figs.png"))