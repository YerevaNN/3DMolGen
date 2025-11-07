import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VQVAE(nn.Module):
    def __init__(self, beta=0.25, n_descriptors=14, n_hidden_dim=128, latent_space_dim=5, struct_vocab_size=256):
        super(VQVAE, self).__init__()
        self.device = torch.device("cpu")
        self.beta = beta
        self.n_descriptors = n_descriptors
        self.n_hidden_dim = n_hidden_dim
        self.latent_space_dim = latent_space_dim
        self.struct_vocab_size = struct_vocab_size
        self.register_buffer("code_book_usage", torch.zeros(self.struct_vocab_size))
        self.encoder = nn.Sequential(
            nn.Linear(self.n_descriptors, self.n_hidden_dim),
            # nn.BatchNorm1d(self.n_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.n_hidden_dim, self.n_hidden_dim),
            # nn.BatchNorm1d(self.n_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.n_hidden_dim, self.latent_space_dim)
        )
        self.embedding = nn.Embedding(self.struct_vocab_size, self.latent_space_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.struct_vocab_size, 1.0 / self.struct_vocab_size)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_space_dim, self.n_hidden_dim),
            # nn.BatchNorm1d(self.n_hidden_dim), use RMSNorm
            nn.ReLU(),
            nn.Linear(self.n_hidden_dim, self.n_descriptors),
        )

    def forward(self, x):
        quant_input = self.encoder(x)
        distances = torch.cdist(quant_input, self.embedding.weight)
        min_idx = torch.argmin(distances, dim=1) 
        min_encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.struct_vocab_size).to(quant_input.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        self.code_book_usage += min_encoding_indices.sum(0)
        quant_out = torch.matmul(min_encodings, self.embedding.weight).view(quant_input.shape)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        code_book_loss = torch.mean((quant_input.detach() - quant_out)**2)
        commitment_loss = torch.mean((quant_input - quant_out.detach())**2)
        quant_loss = code_book_loss + self.beta * commitment_loss
        quant_out = quant_input + (quant_out - quant_input).detach()
        output = self.decoder(quant_out)

        return output, quant_loss, perplexity, min_encodings, min_encoding_indices
    
    def to(self, device):
        # Override the 'to' method to update the device attribute
        self.device = device
        return super(VQVAE, self).to(device)
    
    def print_codebook_utilization(self):
        total_usage = self.code_book_usage.sum().item()
        used_codes = (self.code_book_usage > 0).sum().item()
        utilization = used_codes / self.struct_vocab_size * 100
        print(
            f"Codebook utilization: {utilization:.2f}% ({used_codes}/{self.struct_vocab_size} codes used)"
        )
        return utilization