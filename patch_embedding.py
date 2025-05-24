# patch_embedding.py
import torch
import torch.nn as nn
import numpy as np

class PatchEmbedding(nn.Module):
    """
    DINOv2-inspired patch embedding module for noised images from a diffusion model.
    Converts image to patches, flattens, projects, adds positional embeddings, and prepends CLS token.
    """
    def __init__(self, img_size=224, patch_size=14, in_channels=3, embed_dim=768):
        """
        Initialize the patch embedding module.
        
        Args:
            img_size (int): Input image size (assumed square, e.g., 224).
            patch_size (int): Size of each patch (e.g., 14).
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            embed_dim (int): Dimension of the output embeddings (e.g., 768).
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch splitting and linear projection
        self.patch_dim = in_channels * patch_size * patch_size
        self.projection = nn.Linear(self.patch_dim, embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        """
        Process the input image through patch embedding.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, in_channels, img_size, img_size).
                              Typically a noised image from a diffusion model.
        
        Returns:
            torch.Tensor: Patch embeddings with CLS token, shape (batch_size, num_patches + 1, embed_dim).
        """
        batch_size, _, h, w = x.shape
        if h != self.img_size or w != self.img_size:
            raise ValueError(f"Input image size ({h}x{w}) must match expected size ({self.img_size}x{self.img_size})")

        # 1. Patch splitting
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (batch, h//patch, w//patch, channels, patch, patch)
        x = x.view(batch_size, self.num_patches, -1)    # (batch, num_patches, patch_dim)

        # 2. Linear projection
        x = self.projection(x)  # (batch, num_patches, embed_dim)

        # 3. Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch, num_patches + 1, embed_dim)

        # 4. Add positional embedding
        x = x + self.pos_embed  # (batch, num_patches + 1, embed_dim)

        return x

def process_noised_image(noised_image, img_size=224, patch_size=14, embed_dim=768, in_channels=3):
    """
    Process a noised image through the patch embedding pipeline.
    
    Args:
        noised_image (torch.Tensor): Noised image from diffusion model, shape (batch_size, channels, height, width).
        img_size (int): Target image size (default: 224).
        patch_size (int): Patch size (default: 14).
        embed_dim (int): Embedding dimension (default: 768).
        in_channels (int): Number of input channels (default: 3).
    
    Returns:
        torch.Tensor: Patch embeddings with CLS token, shape (batch_size, num_patches + 1, embed_dim).
    """
    # Initialize the patch embedding module
    patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
    
    # Ensure input is on the correct device
    device = noised_image.device
    patch_embed = patch_embed.to(device)
    
    # Process the image
    with torch.no_grad():
        embeddings = patch_embed(noised_image)
    
    return embeddings