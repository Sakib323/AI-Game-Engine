# world_gen.py

# -*- coding: utf-8 -*-
# World Model Architecture for Interactive Game Generation (Genie-style).
# This model predicts the NEXT frame based on:
# 1. Past Frames (State)
# 2. User Action (Keyboard/Mouse)
# 3. Text Description (Game Rules/Style)

from __future__ import annotations

import math
from typing import Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.utils import logging
from einops import rearrange

# Re-using your optimized kernels
from mmfreelm.layers.hgrn_bit import HGRNBitAttention
from mmfreelm.modules import RMSNorm, LayerNorm
from mmfreelm.ops.bitnet import BitLinear as StandardBitLinear
from mmfreelm.ops.fusedbitnet import FusedBitLinear as FusedBitLinear
from mmfreelm.models.hgrn_bit.video_gen import (
    HGRNBitMLP, 
    FullPrecisionMLP, 
    TimestepEmbedder, 
    TextEmbedder, 
    VideoPatchEmbedder,
    DualCrossAttention, 
    FullPrecisionAdaLNConditioning,
    modulate
)

logger = logging.get_logger(__name__)

#################################################################################
#                           Action Embedding Layer                              #
#################################################################################

class ActionEmbedder(nn.Module):
    """
    Embeds discrete user actions (e.g., arrow keys) into vector representations.
    Mapping (Example):
    0: No-Op
    1: Up, 2: Down, 3: Left, 4: Right
    5: Space (Jump), 6: Click, 7: Interaction
    """
    def __init__(self, num_actions, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_actions + 1, hidden_size)
        self.null_idx = num_actions
        self.dropout_prob = dropout_prob
        # We use a simple MLP to project action embeddings to the model dimension
        self.mlp = FullPrecisionMLP(hidden_size, 4, hidden_act='swish')

    def forward(self, action_ids, train=True, force_drop_ids=None):
        # Randomly drop actions during training to learn "unconditioned" dynamics (physics)
        if (train and self.dropout_prob > 0) or (force_drop_ids is not None):
            if force_drop_ids is None:
                drop_mask = torch.rand(action_ids.shape[0], device=action_ids.device) < self.dropout_prob
            else:
                drop_mask = force_drop_ids == 1
            action_ids = torch.where(drop_mask, self.null_idx, action_ids)

        # Embed and Project
        embs = self.embedding(action_ids)
        return self.mlp(embs).unsqueeze(1) # [B, 1, D]


#################################################################################
#                      World Model Transformer Blocks                           #
#################################################################################

class ActionConditionedDualStreamBlock(nn.Module):
    """
    A Dual-Stream block that attends to both TEXT (Context) and ACTION (Control).
    Stream X: Video Latents (The World)
    Stream C: Text + Action (The Rules & Input)
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, use_rope: bool = False, use_ternary_rope: bool = False, optimized_bitlinear: bool = True, full_precision: bool = False):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm4 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # HGRN Attention for Video Stream
        self.attn_x = HGRNBitAttention(mode='fused_recurrent', hidden_size=hidden_size, num_heads=num_heads, expand_ratio=1, use_short_conv=False, rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        
        # HGRN Attention for Control Stream (Text + Action)
        self.attn_c = HGRNBitAttention(mode='fused_recurrent', hidden_size=hidden_size, num_heads=num_heads, expand_ratio=1, use_short_conv=False, rotary_embeddings=use_rope, use_ternary_rope=use_ternary_rope, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)

        self.dual_cross_attn = DualCrossAttention(hidden_size, num_heads, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)

        self.mlp = HGRNBitMLP(hidden_size, hidden_ratio=mlp_ratio, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)
        self.mlp_c = HGRNBitMLP(hidden_size, hidden_ratio=mlp_ratio, optimized_bitlinear=optimized_bitlinear, full_precision=full_precision)

        self.adaLN_modulation = FullPrecisionAdaLNConditioning(hidden_size, hidden_size, 12 * hidden_size, eps=1e-6, hidden_ratio=mlp_ratio)

    def forward(self, x, c, t):
        # AdaLN Modulation based on timestep t
        modulated_t = ACT2FN['silu'](t)
        params = self.adaLN_modulation(modulated_t)
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x, \
        shift_msa_c, scale_msa_c, gate_msa_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = params.chunk(12, dim=1)

        # --- Video Stream (World State) ---
        modulated_x = modulate(self.norm1(x), shift_msa_x, scale_msa_x)
        attn_x, _, _ = self.attn_x(modulated_x)
        x = x + gate_msa_x.unsqueeze(1) * attn_x

        modulated_x = modulate(self.norm2(x), shift_mlp_x, scale_mlp_x)
        mlp_x = self.mlp(modulated_x)
        x = x + gate_mlp_x.unsqueeze(1) * mlp_x

        # --- Control Stream (Action + Text) ---
        # This stream mixes the game rules (Text) with the user input (Action)
        modulated_c = modulate(self.norm3(c), shift_msa_c, scale_msa_c)
        attn_c, _, _ = self.attn_c(modulated_c)
        c = c + gate_msa_c.unsqueeze(1) * attn_c

        modulated_c = modulate(self.norm4(c), shift_mlp_c, scale_mlp_c)
        mlp_c = self.mlp_c(modulated_c)
        c = c + gate_mlp_c.unsqueeze(1) * mlp_c

        # --- Fusion (Inject Action into World) ---
        x, c = self.dual_cross_attn(x, c)
        return x, c


#################################################################################
#                            WorldDiT Main Class                                #
#################################################################################

class WorldDiT(nn.Module):
    """
    Action-Conditioned Diffusion World Model.
    Generates the NEXT frame based on History + Action + Text.
    """
    def __init__(
        self,
        input_size: Tuple[int, int, int] = (1, 64, 64), # (1 Frame Prediction, H, W)
        patch_size: Tuple[int, int, int] = (1, 2, 2),   # (t, h, w)
        in_channels: int = 4,                           # VAE Latent channels
        vocab_size: int = 49408,
        num_actions: int = 20,                          # Discrete actions supported
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout_prob: float = 0.1,
        num_dual_stream_blocks: int = 14,
        use_rope: bool = False,
        use_ternary_rope: bool = False,
        optimized_bitlinear: bool = True,
        full_precision: bool = False,
        history_frames: int = 4,                        # Number of past frames to condition on
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 
        self.input_size = input_size 
        self.patch_size = patch_size
        self.history_frames = history_frames

        # Token Calculations
        self.t_patches = input_size[0] // patch_size[0]
        self.h_patches = input_size[1] // patch_size[1]
        self.w_patches = input_size[2] // patch_size[2]
        self.num_patches = self.t_patches * self.h_patches * self.w_patches
        self.patch_dim = (patch_size[0] * patch_size[1] * patch_size[2]) * self.out_channels

        # 1. Embedders
        # Main input is the NOISY next frame we are trying to denoise
        self.x_embedder = VideoPatchEmbedder(in_channels, hidden_size, patch_size, optimized_bitlinear, full_precision)
        
        # Condition: History (Past 'k' frames, clean latents)
        # We stack history channel-wise or use a separate encoder. 
        # Here we use a separate encoder for history context.
        self.history_embedder = VideoPatchEmbedder(in_channels, hidden_size, patch_size, optimized_bitlinear, full_precision)
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = TextEmbedder(vocab_size, hidden_size, dropout_prob)
        self.action_embedder = ActionEmbedder(num_actions, hidden_size, dropout_prob)

        # 2. Blocks
        self.dual_stream_blocks = nn.ModuleList([
            ActionConditionedDualStreamBlock(hidden_size, num_heads, mlp_ratio, use_rope, use_ternary_rope, optimized_bitlinear, full_precision)
            for _ in range(num_dual_stream_blocks)
        ])
        
        # Final single stream refinement
        num_single_stream_blocks = depth - num_dual_stream_blocks
        # We reuse the SingleStreamBlock from video_gen (make sure to import it or copy it here)
        from mmfreelm.models.hgrn_bit.video_gen import SingleStreamBlock, FinalLayer
        
        self.single_stream_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, num_heads, mlp_ratio, use_rope, use_ternary_rope, optimized_bitlinear, full_precision)
            for _ in range(num_single_stream_blocks)
        ])

        self.final_layer = FinalLayer(hidden_size, self.patch_dim, mlp_ratio, optimized_bitlinear, full_precision)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, StandardBitLinear, FusedBitLinear, nn.Conv3d, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Zero-out output layers
        for block in self.dual_stream_blocks:
            nn.init.constant_(block.adaLN_modulation.output_proj.weight, 0)
            nn.init.constant_(block.adaLN_modulation.output_proj.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        # Same unpatchify logic as VideoDiT
        b, l, d = x.shape
        c_out = self.out_channels
        pt, ph, pw = self.patch_size
        nt, nh, nw = self.t_patches, self.h_patches, self.w_patches
        x = x.reshape(b, nt, nh, nw, pt, ph, pw, c_out)
        x = torch.einsum('bthwzyxc->bctzhwyx', x)
        x = x.reshape(b, c_out, nt * pt, nh * ph, nw * pw)
        return x

    def forward(self, x, t, y, action_ids, history_frames):
        """
        x: [B, C, 1, H, W] - The Noisy NEXT frame
        t: [B] - Timestep
        y: Dict("input_ids", "attention_mask") - Game Description
        action_ids: [B] - The discrete action taken (0=Left, 1=Right, etc.)
        history_frames: [B, C, K, H, W] - The clean previous K frames (Context)
        """
        # 1. Embed Noisy Target
        x_tokens = self.x_embedder(x) 
        
        # 2. Embed History (Context)
        # We process history frames and concatenate them to the input stream or treat as conditioning
        # Strategy: Concat history tokens to x_tokens initially, but mask them out at loss calculation?
        # Better Strategy for DiT: Treat history as a conditioning stream or prefix tokens.
        # Here we simply add them to the sequence (HGRN handles long seq well)
        hist_tokens = self.history_embedder(history_frames)
        
        # 3. Embed Control (Text + Action)
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y["input_ids"], y["attention_mask"], train=self.training)
        act_emb = self.action_embedder(action_ids, train=self.training)
        
        # Concatenate Text and Action for the "Control Stream"
        # [B, N_text + 1, D]
        c_tokens = torch.cat([y_emb, act_emb], dim=1)

        # 4. Dual Stream Processing
        # We pass (History + NoisyFrame) as the visual stream
        visual_tokens = torch.cat([hist_tokens, x_tokens], dim=1)
        
        for block in self.dual_stream_blocks:
            visual_tokens, c_tokens = block(visual_tokens, c_tokens, t_emb)

        # 5. Single Stream Processing
        combined_tokens = torch.cat([visual_tokens, c_tokens], dim=1)
        
        for block in self.single_stream_blocks:
            combined_tokens = block(combined_tokens, t_emb)

        # 6. Output Projection
        # We only want to predict the noise for the 'x' part (the last tokens of the visual stream)
        # history tokens are ignored for output
        num_hist = hist_tokens.shape[1]
        num_x = x_tokens.shape[1]
        
        # Extract just the x_tokens part from the combined sequence
        # Structure: [History | Target | Text | Action]
        target_tokens = combined_tokens[:, num_hist : num_hist + num_x]
        
        output = self.final_layer(target_tokens, t_emb)
        return self.unpatchify(output)

    def predict_next_frame(self, last_frames, action_id, prompt_ids, cfg_scale=5.0):
        """
        Inference function for the Game Loop.
        last_frames: Clean latent tensor of history
        action_id: Integer
        """
        self.eval()
        with torch.no_grad():
            # Start from random noise for the next frame
            x_t = torch.randn_like(last_frames[:, :, -1:]).to(last_frames.device) # [B, C, 1, H, W]
            
            # Standard Diffusion Sampling Loop (DDIM/DDPM) would go here
            # This is a simplified pseudo-code for the loop
            for t in self.scheduler.timesteps:
                # Expand inputs for CFG (Uncond, ActionCond, TextCond) if needed
                # ...
                pass
                
            return x_t # The denoised next frame

#################################################################################
#                                 Configs                                       #
#################################################################################

def WorldDiT_S(**kwargs):
    return WorldDiT(depth=12, hidden_size=384, num_heads=6, num_dual_stream_blocks=6, **kwargs)

def WorldDiT_B(**kwargs):
    return WorldDiT(depth=12, hidden_size=768, num_heads=12, num_dual_stream_blocks=6, **kwargs)

def WorldDiT_XL(**kwargs):
    return WorldDiT(depth=28, hidden_size=1152, num_heads=16, num_dual_stream_blocks=14, **kwargs)

WorldDiT_models = {
    'WorldDiT-S': WorldDiT_S,
    'WorldDiT-B': WorldDiT_B,
    'WorldDiT-XL': WorldDiT_XL,
}