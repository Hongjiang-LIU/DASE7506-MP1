"""Single-device, shape-batched Muon for training hidden matrices only.

Algorithm reference: Keller Jordan's MIT-licensed Muon implementation,
https://github.com/KellerJordan/Muon/blob/master/muon.py . Copyright (c) 2024
Keller Jordan; the complete license is in THIRD_PARTY_MUON_LICENSE.txt.

This local implementation uses the reference momentum, Nesterov interpolation,
five-step quintic Newton-Schulz coefficients, and rectangular update scaling.
Differences are explicit: same-shaped tensors are stacked; input gradients are
never changed; absent gradients are skipped; CPU NS uses FP32 rather than BF16.
CUDA NS uses BF16. There is no distributed behavior, compiler requirement,
corpus access, model change, or inference-time use of this optimizer.
"""
import math

import torch


def partition_named_parameters(model):
    """Return (Muon named parameters, auxiliary AdamW named parameters).

    Only trainable 2-D matrices whose names start with blocks. belong to Muon.
    In particular token embeddings, a tied/untied output head, normalization
    gains, pointer weights, and scalar gates remain the driver's AdamW job.
    named_parameters deduplicates the tied token/head Parameter automatically.
    """
    hidden, auxiliary = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        selected = name.startswith("blocks.") and parameter.ndim == 2
        (hidden if selected else auxiliary).append((name, parameter))
    return hidden, auxiliary


@torch.no_grad()
def zeropower_via_newtonschulz5(matrix, steps=5):
    """Reference quintic iteration on a matrix or a batch of matrices.

    Finite iteration approximates a zeroth-power update; it is not an exact
    polar/SVD factor. No in-place operations touch the supplied matrix.
    """
    if matrix.ndim < 2 or min(matrix.shape[-2:]) == 0 or not matrix.is_floating_point():
        raise ValueError("Newton-Schulz input must contain nonempty floating-point matrices")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("Newton-Schulz steps must be a positive integer")
    if matrix.device.type not in ("cpu", "cuda"):
        raise ValueError("This training optimizer supports CPU or CUDA only")
    dtype = torch.bfloat16 if matrix.device.type == "cuda" else torch.float32
    # Explicitly disable an enclosing autocast context: dtype choice is fixed.
    with torch.autocast(device_type=matrix.device.type, enabled=False):
        x = matrix.to(dtype=dtype)
        transposed = matrix.shape[-2] > matrix.shape[-1]
        if transposed:
            x = x.mT
        x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(steps):
            gram = x @ x.mT
            # Preserve the reference expression's multiplication order.
            polynomial = b * gram + (c * gram) @ gram
            x = a * x + polynomial @ x
        return x.mT if transposed else x


def _validate_group(group):
    for name in ("lr", "base_lr", "weight_decay"):
        value = group[name]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    beta = group["momentum"]
    if not isinstance(beta, (int, float)) or not math.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("momentum must be in [0,1)")
    if not isinstance(group["nesterov"], bool):
        raise ValueError("nesterov must be boolean")
    steps = group["ns_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("ns_steps must be a positive integer")


class BatchedMuon(torch.optim.Optimizer):
    """Muon-only optimizer; pass the hidden list from partition_named_parameters.

    The driver owns a separate AdamW for all remaining parameters and must
    update both optimizers' learning rates when scheduling. base_lr is retained
    per parameter group and in state_dict for the existing cosine scheduler.
    A None gradient causes no parameter decay, update, or state initialization.
    """
    def __init__(self, params, lr=.002, weight_decay=0., momentum=.95,
                 ns_steps=5, nesterov=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        ns_steps=ns_steps, nesterov=nesterov)
        super().__init__(params, defaults)
        for group in self.param_groups:
            group.setdefault("base_lr", group["lr"])
            _validate_group(group)
            for parameter in group["params"]:
                if parameter.ndim != 2 or not parameter.is_floating_point():
                    raise ValueError("Muon accepts only floating-point 2-D hidden matrices")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            _validate_group(group)
            buckets = {}
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                if parameter.ndim != 2 or gradient.shape != parameter.shape:
                    raise ValueError("Muon requires matrix-shaped dense gradients")
                key = (parameter.device, parameter.dtype, tuple(parameter.shape))
                buckets.setdefault(key, []).append(parameter)
            for parameters in buckets.values():
                gradient_batch = torch.stack([parameter.grad for parameter in parameters])
                buffers = [self.state[parameter].get("momentum_buffer") for parameter in parameters]
                if all(buffer is None for buffer in buffers):
                    momentum_batch = torch.zeros_like(gradient_batch)
                else:
                    momentum_batch = torch.stack([
                        torch.zeros_like(parameter) if buffer is None else buffer
                        for parameter, buffer in zip(parameters, buffers)])
                beta = group["momentum"]
                momentum_batch.lerp_(gradient_batch, 1 - beta)
                # Out-of-place lerp deliberately avoids mutating original p.grad.
                direction = (gradient_batch.lerp(momentum_batch, beta)
                             if group["nesterov"] else momentum_batch)
                update = zeropower_via_newtonschulz5(direction, group["ns_steps"])
                rows, columns = parameters[0].shape
                update = update * math.sqrt(max(1., rows / columns))
                # The model and optimizer state retain the parameter dtype;
                # CUDA's temporary NS calculations alone use BF16.
                update = update.to(dtype=parameters[0].dtype)
                if group["weight_decay"]:
                    torch._foreach_mul_(parameters, 1 - group["lr"] * group["weight_decay"])
                torch._foreach_add_(parameters, list(update.unbind(0)), alpha=-group["lr"])
                for index, parameter in enumerate(parameters):
                    state = self.state[parameter]
                    # Each view is disjoint and shares only this group's new
                    # backing storage. Normal Optimizer state serialization works.
                    state["momentum_buffer"] = momentum_batch[index]
                    state["step"] = state.get("step", 0) + 1
        return loss


# Concise import name for training drivers.
Muon = BatchedMuon
