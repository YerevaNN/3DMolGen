import torch
import torch.nn as nn
import numpy as np
import aim
from vqvae import VQVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 20
batch_size = 32


model = VQVAE().to(device)
data = np.random.random((1000, 14))
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(epochs):
    for i in range(0, data.shape[0]+ batch_size, batch_size):
        batch = torch.Tensor(data[i:i+batch_size,:]).to(model.device)
        optimizer.zero_grad()
        out, quant_loss = model(batch)
        reconstruction_loss = criterion(out, batch)
        loss = reconstruction_loss + quant_loss
        loss.backward()
        optimizer.step()
        print(f'{epoch=}, {loss.item()=}, {quant_loss.item()=}, {reconstruction_loss.item()=}')