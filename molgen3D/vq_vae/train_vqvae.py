import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import aim
from vqvae import VQVAE


def train(model, dataloader, optimizer, criterion, num_epochs, device, aim):
    model.to(device)
    print(f'model is on {model.device=}')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for en, batch in enumerate(dataloader):
            optimizer.zero_grad()
            out, quant_loss, perplexity, _, _ = model(batch)
            reconstruction_loss = criterion(out, batch)
            loss = reconstruction_loss + quant_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            aim.track(reconstruction_loss, name="reconstruction_loss")
            aim.track(quant_loss, name="quant_loss")
            aim.track(perplexity, name="perplexity")
            aim.track(loss, name="overall_loss")
            # if en  % 1000 == 0:
            #     print(f'{epoch=}, {en=}, {loss.item()=}, {quant_loss.item()=}, {reconstruction_loss.item()=}')

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        codebook_utilization = model.print_codebook_utilization()
        aim.track(avg_loss, name='avg_loss', epoch=epoch)
        aim.track(codebook_utilization, name='codebook_utilization', epoch=epoch)

class NumpyDataset(Dataset):
    def __init__(self, npy_file, device):
        self.data = np.load(npy_file)
        self.device = device    
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]).to(self.device)

def get_dataloader(path, batch_size, device):
    dataset = NumpyDataset(path, device)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Adjust based on your CPU cores
        pin_memory=False  # Use if training on GPU
    )
    return dataloader

def main(args):
    aim_run = aim.Session()
    aim_run.set_params(args)    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = VQVAE(
        beta=args.beta, 
        n_descriptors=args.num_descriptors, 
        n_hidden_dim=args.hidden_dim, 
        latent_space_dim=args.latent_dim, 
        struct_vocab_size=args.num_embeddings
    )

    dataloader = get_dataloader(args.data_path, args.batch_size, device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()

    train(model, dataloader, optimizer, criterion, args.num_epochs, device, aim_run)

    torch.save(
        model.state_dict(),
        f"vqvae_mol_coordinates.pth",
    )
    print("Training completed and model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQVAE atom coordinates dataset")
    parser.add_argument("--num_descriptors", type=int, default=3, help="Number of descriptors")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Number of warmup epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--latent_dim", type=int, default=5, help="Latent dimension")
    parser.add_argument(
        "--num_embeddings", type=int, default=256, help="Number of embeddings for VQ"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Hidden layers dimension"
    )
    parser.add_argument("--data_path", type=str, help="Path to read data from")
    parser.add_argument("--beta", type=float, default=0.2, help="Commitment loss beta")

    args = parser.parse_args()
    main(args)