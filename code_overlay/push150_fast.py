"""FP32 CPU implementation variants of the same push150 trained predictor.

No learned tensor is added, no input/result is memoized, and state_dict remains
identical to push150_expand. The optional fused dense gate/value matrix is only
a nonpersistent concatenation of current train-derived weights. It contributes
to measured runtime RAM, not a second checkpoint. FP32 operation fusion can
change rounding, so equality is numerical rather than necessarily bitwise.

The optimized path is CPU + evaluation + no-grad only. GPU, training, and any
gradient-enabled call retain the original differentiable features/cache path.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from push150_expand import ExpandedSignedCache


def pair_rotary(x, rotary):
    """Adjacent real pairs: (a*cos-b*sin, b*cos+a*sin), all in FP32."""
    length = x.shape[-2]
    cosine, sine = rotary.cos[:, :, :length, ::2], rotary.sin[:, :, :length, ::2]
    even, odd = x[..., ::2], x[..., 1::2]
    result = torch.empty_like(x)
    result[..., ::2] = torch.addcmul(even*cosine, odd, sine, value=-1)
    result[..., 1::2] = torch.addcmul(odd*cosine, even, sine)
    return result


def rms_norm_no_grad(x, norm):
    # Never overwrite x: it is also the residual connection input.
    scale = x.float().pow(2).mean(dim=-1, keepdim=True).add(norm.eps).rsqrt()
    result = x*scale.to(dtype=x.dtype)
    return result.mul_(norm.weight)


class FastExpandedSignedCache(ExpandedSignedCache):
    def __init__(self, config):
        super().__init__(config)
        size = self.config.get('cpu_feature_microbatch', 32)
        if isinstance(size, bool) or not isinstance(size, int) or size not in (8,16,32):
            raise ValueError('cpu_feature_microbatch must be exactly 8, 16 or 32')
        self.config['cpu_feature_microbatch'] = size
        for name in ('push150_fast_fuse_dense_mlp', 'push150_fast_paired_rope', 'push150_fast_inplace_rms'):
            value = self.config.get(name, True)
            if not isinstance(value, bool):
                raise ValueError(f'{name} must be a boolean')
            self.config[name] = value
        self._fused_dense_weights = {}
        self.register_load_state_dict_post_hook(lambda module, incompatible: module._clear_fused_weights())

    def _clear_fused_weights(self):
        self._fused_dense_weights = {}

    def _apply(self, function, recurse=True):
        self._clear_fused_weights()
        result = super()._apply(function, recurse=recurse)
        self._clear_fused_weights()
        return result

    def train(self, mode=True):
        if mode:
            self._clear_fused_weights()
        return super().train(mode)

    def _fast_enabled(self, tensor):
        return (not self.training and not torch.is_grad_enabled() and
                tensor.device.type == 'cpu' and self.token.weight.dtype == torch.float32 and
                not torch.is_autocast_enabled('cpu'))

    def _dense_gate_value_weight(self, index, gate, value):
        # _version detects ordinary in-place optimizer/parameter updates even
        # when an external caller forgets to switch from eval to train first.
        try:
            signature = tuple((id(weight), weight._version, weight.data_ptr(),
                               tuple(weight.shape), weight.dtype, weight.device)
                              for weight in (gate.weight, value.weight))
        except RuntimeError:
            # Inference tensors without version counters cannot be safely
            # retained across calls; concatenate anew rather than risk staleness.
            return torch.cat((gate.weight, value.weight), dim=0)
        cached = self._fused_dense_weights.get(index)
        if cached is None or cached[0] != signature:
            weight = torch.cat((gate.weight, value.weight), dim=0)
            self._fused_dense_weights[index] = (signature, weight)
        return self._fused_dense_weights[index][1]

    def _normalization(self, x, norm):
        return rms_norm_no_grad(x, norm) if self.config['push150_fast_inplace_rms'] else norm(x)

    @torch.no_grad()
    def _features_part(self, ids):
        x = self.token(ids)
        if self.pos is not None:
            x.add_(self.pos(torch.arange(ids.shape[1], device=ids.device)))
        for index, block in enumerate(self.blocks):
            batch, length, width = x.shape
            q, k, v = block.qkv(self._normalization(x, block.norm1)).view(
                batch, length, 3, block.heads, width//block.heads).permute(2,0,3,1,4)
            if block.rotary is not None:
                if self.config['push150_fast_paired_rope']:
                    q, k = pair_rotary(q, block.rotary), pair_rotary(k, block.rotary)
                else:
                    q, k = block.rotary(q), block.rotary(k)
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            branch = block.proj(attended.transpose(1,2).reshape(batch,length,width))
            x.add_(branch)
            normalized = self._normalization(x, block.norm2)
            gate, value = block.mlp.gate, block.mlp.value
            if (self.config['push150_fast_fuse_dense_mlp'] and isinstance(gate, nn.Linear) and
                    isinstance(value, nn.Linear) and gate.bias is None and value.bias is None):
                weight = self._dense_gate_value_weight(index, gate, value)
                gate_output, value_output = F.linear(normalized, weight).chunk(2, dim=-1)
                F.silu(gate_output, inplace=True).mul_(value_output)
                branch = block.mlp.out(gate_output)
            else:
                branch = block.mlp(normalized)
            x.add_(branch)
        return self._normalization(x, self.norm)

    def features(self, ids):
        if not self._fast_enabled(ids):
            return super().features(ids)
        size = self.config['cpu_feature_microbatch']
        if ids.shape[0] <= size:
            return self._features_part(ids)
        return torch.cat([self._features_part(part) for part in ids.split(size)], dim=0)

    def cache_attention_and_gate(self, features):
        if not self._fast_enabled(features):
            return super().cache_attention_and_gate(features)
        _, length, _ = features.shape
        normalized = F.normalize(features.float(), dim=-1)
        theta, alpha = self.cache_scales()  # Preserve signed alpha and theta bounds.
        scores = normalized@normalized.transpose(-1,-2)
        scores.mul_(theta)
        query, key = self.pointer_query(normalized).float(), self.pointer_key(normalized).float()
        residual = query@key.transpose(-1,-2)
        residual.div_(math.sqrt(query.shape[-1]))
        scores.add_(residual)
        scores.sub_(alpha*self._cache_log_distance[:length,:length])
        scores.masked_fill_(~self._cache_allowed[:length,:length], -1e4)
        attention = scores.softmax(-1)
        position = torch.arange(length, device=features.device, dtype=torch.float32)
        history = (position/max(1,self.context-1))[None,:].expand(features.shape[0],-1)
        gate_input = torch.cat((normalized, attention.amax(-1)[...,None],
                               attention.square().sum(-1)[...,None], history[...,None]), dim=-1)
        gate = self.pointer_gate(gate_input).float()
        gate.sigmoid_().mul_(self.config.get('pointer_max_weight', .5))
        gate.mul_((position > 0)[None,:,None])
        return attention, gate


def build_model(config):
    return FastExpandedSignedCache(config)
