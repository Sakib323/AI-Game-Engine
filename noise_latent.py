import math
import numpy as np
import torch as th

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    Copied from gaussian_diffusion.py for consistency.
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "squaredcos_cap_v2":
        def alpha_bar(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        betas = np.array(betas)
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")
    return betas

class NoiseLatent:
    """
    A class to introduce Gaussian noise into latent representations for DiT architectures.
    Generates noisy latents x_t from x_0 using the forward diffusion process q(x_t | x_0).
    """
    def __init__(
        self,
        num_timesteps=1000,
        beta_schedule="linear",
        device="cuda" if th.cuda.is_available() else "cpu"
    ):
        """
        Initialize the NoiseLatent class with a beta schedule for noise levels.
        
        Args:
            num_timesteps (int): Number of diffusion timesteps (default: 1000).
            beta_schedule (str): Type of beta schedule ("linear" or "squaredcos_cap_v2").
            device (str): Device to perform computations on ("cuda" or "cpu").
        """
        self.num_timesteps = num_timesteps
        self.device = device

        # Get beta schedule
        betas = get_named_beta_schedule(beta_schedule, num_timesteps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "Betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        # Precompute diffusion parameters
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        # Move precomputed arrays to torch tensors on the specified device
        self.sqrt_alphas_cumprod = th.from_numpy(self.sqrt_alphas_cumprod).float().to(device)
        self.sqrt_one_minus_alphas_cumprod = th.from_numpy(self.sqrt_one_minus_alphas_cumprod).float().to(device)

    def add_noise(self, x_start, t, noise=None):
        """
        Add Gaussian noise to the input latent to sample x_t from q(x_t | x_0).
        
        Args:
            x_start (torch.Tensor): Input latent tensor [batch, channels, height, width].
            t (torch.Tensor): Timestep indices [batch], values in [0, num_timesteps-1].
            noise (torch.Tensor, optional): Specific Gaussian noise to use. If None, randomly generated.
        
        Returns:
            torch.Tensor: Noisy latent x_t with the same shape as x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start).to(self.device)
        assert noise.shape == x_start.shape, f"Noise shape {noise.shape} must match input shape {x_start.shape}"
        assert t.shape[0] == x_start.shape[0], f"Timestep batch size {t.shape[0]} must match input batch size {x_start.shape[0]}"

        # Extract coefficients for the given timesteps
        sqrt_alphas_cumprod_t = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        # Compute noisy latent: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
        noisy_latent = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_latent

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D tensor for a batch of indices.
        
        Args:
            arr (torch.Tensor): 1-D tensor of precomputed values.
            timesteps (torch.Tensor): Tensor of indices to extract.
            broadcast_shape (tuple): Desired output shape with batch dim matching timesteps.
        
        Returns:
            torch.Tensor: Tensor of shape [batch_size, 1, ...] matching broadcast_shape.
        """
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)