# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM, HGRNBitModel
from mmfreelm.models.hgrn_bit.rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb, rotate_half
from mmfreelm.models.hgrn_bit.ternary_dit import TimestepEmbedder, DiTBlock, TextEmbedder, FinalLayer, DiT_models
from mmfreelm.models.hgrn_bit.mesh_dit import MeshDiT_models

AutoConfig.register(HGRNBitConfig.model_type, HGRNBitConfig)
AutoModel.register(HGRNBitConfig, HGRNBitModel)
AutoModelForCausalLM.register(HGRNBitConfig, HGRNBitForCausalLM)

__all__ = [
    'HGRNBitConfig',
    'HGRNBitForCausalLM',
    'HGRNBitModel',
    'RotaryEmbedding',
    'apply_rotary_pos_emb',
    'rotate_half',
    'TimestepEmbedder',
    'DiTBlock',
    'TextEmbedder',
    'FinalLayer',
    'FinalLayerSecond',
    'DiT_models',
    'MeshDiT_models',
]