import math
import numpy as np
import torch as th
import enum
from tqdm.auto import tqdm

# [NEW] RMSNorm for patch embedding compatibility
class RMSNorm(th.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = th.nn.Parameter(th.ones(dim))
    
    def forward(self, x):
        # x: [batch, seq_len, dim] or [batch, channels, height, width]
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * th.rsqrt(variance + self.eps)
        return x * self.scale

# [NEW] Timestep samplers
class UniformSampler:
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
    
    def sample(self, batch_size, device):
        return th.randint(0, self.num_timesteps, (batch_size,), device=device)

class LossSecondMomentResampler:
    def __init__(self, num_timesteps, history_per_term=10):
        self.num_timesteps = num_timesteps
        self.history_per_term = history_per_term
        self.losses = {i: [] for i in range(num_timesteps)}
    
    def add_loss(self, t, loss):
        self.losses[t.item()].append(loss.item())
        if len(self.losses[t.item()]) > self.history_per_term:
            self.losses[t.item()].pop(0)
    
    def sample(self, batch_size, device):
        weights = []
        for t in range(self.num_timesteps):
            losses = self.losses[t]
            weights.append(np.mean(losses) ** 2 if losses else 1.0)
        weights = np.array(weights)
        weights /= weights.sum()
        indices = np.random.choice(self.num_timesteps, size=batch_size, p=weights)
        return th.tensor(indices, device=device, dtype=th.long)

class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()

class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (-1.0 + logvar2 - logvar1 + th.exp(logvar1 - logvar2) +
                  ((mean1 - mean2) ** 2) * th.exp(-logvar2))

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = th.distributions.Normal(0, 1).cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = th.distributions.Normal(0, 1).cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        centered_x < -0.999,
        log_cdf_plus,
        th.where(centered_x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    return log_probs

def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "quad":
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    return betas

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        def alpha_bar(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return np.array(betas)
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")

def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"Cannot create exactly {desired_count} steps with an integer stride")
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"Cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)

class NoiseLatent:
    """
    A comprehensive diffusion model for adding noise and denoising latent representations in DiT architectures.
    Supports forward diffusion q(x_t | x_0), sampling, DDIM, spaced diffusion, and conditioning.
    """
    def __init__(
        self,
        num_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",  # [CHANGED] Default to squaredcos_cap_v2
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_LARGE,
        loss_type=LossType.MSE,
        device="cuda" if th.cuda.is_available() else "cpu",
        use_timesteps=None,
        timestep_sampler="loss_aware"  # [NEW] Timestep sampler option
    ):
        """
        Initialize the diffusion model with beta schedule, model configurations, and timestep sampling.
        
        Args:
            num_timesteps (int): Number of diffusion timesteps (default: 1000).
            beta_schedule (str): Type of beta schedule (default: "squaredcos_cap_v2").
            model_mean_type (ModelMeanType): Type of model output (default: EPSILON).
            model_var_type (ModelVarType): Variance type (default: FIXED_LARGE).
            loss_type (LossType): Loss function type (default: MSE).
            device (str): Device for computations ("cuda" or "cpu").
            use_timesteps (set, optional): Subset of timesteps for spaced diffusion.
            timestep_sampler (str): Sampler type ("uniform" or "loss_aware").
        """
        self.num_timesteps = num_timesteps
        self.device = device
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        # [NEW] Initialize timestep sampler
        if timestep_sampler == "uniform":
            self.timestep_sampler = UniformSampler(num_timesteps)
        elif timestep_sampler == "loss_aware":
            self.timestep_sampler = LossSecondMomentResampler(num_timesteps)
        else:
            raise ValueError(f"Unknown timestep sampler: {timestep_sampler}")

        # [NEW] Optional RMSNorm for patch embeddings
        self.rms_norm = RMSNorm(dim=768).to(device) if beta_schedule in ["squaredcos_cap_v2", "linear"] else None  # Example dim for ViT

        # Get beta schedule
        betas = get_named_beta_schedule(beta_schedule, num_timesteps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "Betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        # Spaced diffusion setup
        self.use_timesteps = set(range(num_timesteps)) if use_timesteps is None else set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = num_timesteps
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(np.cumprod(1.0 - betas)):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        if use_timesteps is not None:
            betas = np.array(new_betas)

        # Precompute diffusion parameters
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior calculations
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:])) if len(self.posterior_variance) > 1 else np.array([])
        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)

        # Move to device
        for attr in ['sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'log_one_minus_alphas_cumprod',
                     'sqrt_recip_alphas_cumprod', 'sqrt_recipm1_alphas_cumprod', 'posterior_variance',
                     'posterior_log_variance_clipped', 'posterior_mean_coef1', 'posterior_mean_coef2', 'betas']:
            setattr(self, attr, th.from_numpy(getattr(self, attr)).float().to(device))

    def add_noise(self, x_start, t, noise=None):
        """
        Add Gaussian noise to the input latent to sample x_t from q(x_t | x_0).
        
        Args:
            x_start (torch.Tensor): Input latent tensor [batch, channels, height, width] or [batch, seq_len, dim].
            t (torch.Tensor): Timestep indices [batch], values in [0, num_timesteps-1].
            noise (torch.Tensor, optional): Specific Gaussian noise to use.
        
        Returns:
            torch.Tensor: Noisy latent x_t.
        """
        # [NEW] Apply RMSNorm if enabled
        if self.rms_norm is not None:
            x_start = self.rms_norm(x_start)
        
        if noise is None:
            noise = th.randn_like(x_start).to(self.device)
        assert noise.shape == x_start.shape
        assert t.shape[0] == x_start.shape[0]
        sqrt_alphas_cumprod_t = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_sample(self, x_start, t, noise=None):
        """Alias for add_noise for compatibility."""
        return self.add_noise(x_start, t, noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0).
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, clip_denoised=True, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t) and predict x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.shape[:2]
        assert t.shape == (B,)
        t_mapped = th.tensor(self.timestep_map, device=t.device, dtype=t.dtype)[t]
        # [NEW] Apply RMSNorm if enabled
        if self.rms_norm is not None:
            x = self.rms_norm(x)
        model_output = model(x, t_mapped, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            min_log = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = self._extract_into_tensor(th.log(self.betas), t, x.shape)
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    th.from_numpy(np.append(self.posterior_variance.cpu().numpy()[1], self.betas.cpu().numpy()[1:])).to(self.device),
                    th.from_numpy(np.log(np.append(self.posterior_variance.cpu().numpy()[1], self.betas.cpu().numpy()[1:]))).to(self.device),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = self._extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    # [NEW] Conditioning methods
    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a conditional gradient function.
        """
        if model_kwargs is None:
            model_kwargs = {}
        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute conditioned p_mean_variance using the score function.
        """
        if model_kwargs is None:
            model_kwargs = {}
        alpha_bar = self._extract_into_tensor(self.alphas_cumprod, t, x.shape)
        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)
        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(self, model, x, t, clip_denoised=True, model_kwargs=None, cond_fn=None):
        """
        Sample x_{t-1} from the model at the given timestep.
        """
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        noise = th.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, model_kwargs=None, cond_fn=None, device=None, progress=False):
        """
        Generate samples from the model.
        """
        if device is None:
            device = self.device
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(self.use_timesteps)[::-1]

        if progress:
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(model, img, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs, cond_fn=cond_fn)
                img = out["sample"]
        return img

    def ddim_sample(self, model, x, t, clip_denoised=True, model_kwargs=None, cond_fn=None, eta=0.0):
        """
        Sample x_{t-1} using DDIM.
        """
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = self._extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = self._extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        noise = th.randn_like(x)
        mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(self, model, shape, noise=None, clip_denoised=True, model_kwargs=None, cond_fn=None, device=None, progress=False, eta=0.0):
        """
        Generate samples using DDIM.
        """
        if device is None:
            device = self.device
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(self.use_timesteps)[::-1]

        if progress:
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(model, img, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs, cond_fn=cond_fn, eta=eta)
                img = out["sample"]
        return img

    # [NEW] DDIM reverse sampling
    def ddim_reverse_sample(self, model, x, t, clip_denoised=True, model_kwargs=None, cond_fn=None, eta=0.0):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(model, x, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        eps = (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x -
            out["pred_xstart"]
        ) / self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = self._extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
        mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_next) + th.sqrt(1 - alpha_bar_next) * eps
        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample_loop(self, model, x_start, clip_denoised=True, model_kwargs=None, cond_fn=None, device=None, progress=False):
        """
        Encode clean latents into noisy ones using DDIM reverse ODE.
        """
        if device is None:
            device = self.device
        img = x_start
        indices = list(self.use_timesteps)

        if progress:
            indices = tqdm(indices)

        for i in indices[:-1]:  # Skip last timestep
            t = th.tensor([i] * x_start.shape[0], device=device)
            with th.no_grad():
                out = self.ddim_reverse_sample(model, img, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs, cond_fn=cond_fn)
                img = out["sample"]
        return img

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        terms = {}

        if self.loss_type in [LossType.KL, LossType.RESCALED_KL]:
            terms["loss"] = self._vb_terms_bpd(model, x_start, x_t, t, model_kwargs=model_kwargs)["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= len(self.use_timesteps)
        else:
            model_output = model(x_t, th.tensor(self.timestep_map, device=t.device, dtype=t.dtype)[t], **model_kwargs)
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape
            terms["mse"] = mean_flat((target - model_output) ** 2)
            terms["loss"] = terms["mse"]
        
        # [NEW] Update loss for LossSecondMomentResampler
        if isinstance(self.timestep_sampler, LossSecondMomentResampler):
            self.timestep_sampler.add_loss(t[0], terms["loss"].mean())
        
        return terms

    def sample_timesteps(self, batch_size, device):
        """
        Sample timesteps using the configured timestep sampler.
        """
        return self.timestep_sampler.sample(batch_size, device)

    def _vb_terms_bpd(self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)
        decoder_nll = -discretized_gaussian_log_likelihood(x_start, means=out["mean"], log_scales=0.5 * out["log_variance"])
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)