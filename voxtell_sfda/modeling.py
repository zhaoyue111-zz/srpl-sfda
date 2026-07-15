from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def add_voxtell_to_path(voxtell_root: Path) -> None:
    voxtell_root = voxtell_root.resolve()
    if not voxtell_root.is_dir():
        raise FileNotFoundError(f"VoxTell root does not exist: {voxtell_root}")
    root_str = str(voxtell_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def load_voxtell_predictor(voxtell_root: Path, model_dir: Path, device: torch.device):
    add_voxtell_to_path(voxtell_root)
    from voxtell.inference.predictor_multiclass import VoxTellPredictor

    return VoxTellPredictor(model_dir=str(model_dir), device=device)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(base.weight.new_zeros((rank, base.in_features)))
        self.lora_B = nn.Parameter(base.weight.new_zeros((base.out_features, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return result + update * self.scaling

    @property
    def weight(self) -> torch.Tensor:  # torch.nn.MultiheadAttention对out_proj不会走out_proj(x),而是在内部直接访问：self.out_proj.weight和self.out_proj.bias
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias


class LoRAMultiheadAttention(nn.Module):
    """LoRA wrapper for PyTorch MultiheadAttention Q/K/V/O matrices."""

    def __init__(
        self,
        base: nn.MultiheadAttention,
        rank: int,
        alpha: float,
        dropout: float,
        matrices: str,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if base._qkv_same_embed_dim is not True:
            raise ValueError("LoRAMultiheadAttention currently expects packed QKV weights.")
        if dropout != 0:   # linear lora：y=x@W_base+ dropout(x)@A@B*scale,所以dropout可以直接加在输入上，但是attention的lora被合并到了权重中：ΔW = B @ A * scale，没有独立的dropout(x)@A@B分支
            raise ValueError("Attention LoRA currently requires --lora-dropout 0.")
        selected = set(matrices.lower())
        invalid = selected.difference({"q", "k", "v", "o"})
        if invalid:
            raise ValueError(f"Unknown LoRA attention matrix letters: {sorted(invalid)}")
        if not selected:
            raise ValueError("--lora-attention must select at least one of q/k/v/o.")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.matrices = "".join(letter for letter in "qkvo" if letter in selected)

        embed_dim = base.embed_dim
        for param in self.base.parameters():
            param.requires_grad = False
        for letter in self.matrices:
            self.register_parameter(
                f"lora_{letter}_A",
                nn.Parameter(base.in_proj_weight.new_empty((rank, embed_dim))),
            )
            self.register_parameter(
                f"lora_{letter}_B",
                nn.Parameter(base.in_proj_weight.new_zeros((embed_dim, rank))),
            )
            nn.init.kaiming_uniform_(getattr(self, f"lora_{letter}_A"), a=5 ** 0.5)

    def _delta_weight(self, letter: str) -> torch.Tensor:
        if letter not in self.matrices:
            raise KeyError(letter)
        lora_a = getattr(self, f"lora_{letter}_A")
        lora_b = getattr(self, f"lora_{letter}_B")
        return (lora_b @ lora_a) * self.scaling

    def _merged_in_proj_weight(self) -> torch.Tensor:
        weight = self.base.in_proj_weight
        if not any(letter in self.matrices for letter in "qkv"):
            return weight
        delta = torch.zeros_like(weight)
        embed_dim = self.base.embed_dim
        if "q" in self.matrices:
            delta[0:embed_dim, :] = self._delta_weight("q").to(dtype=weight.dtype, device=weight.device)
        if "k" in self.matrices:
            delta[embed_dim : 2 * embed_dim, :] = self._delta_weight("k").to(dtype=weight.dtype, device=weight.device)
        if "v" in self.matrices:
            delta[2 * embed_dim : 3 * embed_dim, :] = self._delta_weight("v").to(dtype=weight.dtype, device=weight.device)
        return weight + delta

    def _merged_out_proj_weight(self) -> torch.Tensor:
        weight = self.base.out_proj.weight
        if "o" not in self.matrices:
            return weight
        return weight + self._delta_weight("o").to(dtype=weight.dtype, device=weight.device)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        is_batched = query.dim() == 3
        if self.base.batch_first and is_batched:
            query, key, value = (x.transpose(1, 0) for x in (query, key, value))
        out, weights = F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.base.embed_dim,
            num_heads=self.base.num_heads,
            in_proj_weight=self._merged_in_proj_weight(),
            in_proj_bias=self.base.in_proj_bias,
            bias_k=self.base.bias_k,
            bias_v=self.base.bias_v,
            add_zero_attn=self.base.add_zero_attn,
            dropout_p=self.base.dropout if self.training else 0.0,
            out_proj_weight=self._merged_out_proj_weight(),
            out_proj_bias=self.base.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        if self.base.batch_first and is_batched:
            out = out.transpose(1, 0)
        return out, weights

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = super().__getattr__("base")
            return getattr(base, name)


def _lora_target_specs(network: nn.Module, target: str) -> list[tuple[nn.Module | None, str | None, nn.Module]]:
    if target == "attention":
        return [
            (network, "transformer_decoder", network.transformer_decoder),
        ]
    if target == "prompt_decoder":
        return [
            (network, "transformer_decoder", network.transformer_decoder),
            (network, "project_bottleneck_embed", network.project_bottleneck_embed),
            (network, "project_text_embed", network.project_text_embed),
            (network, "project_to_decoder_channels", network.project_to_decoder_channels),
        ]
    if target == "decoder":
        return [
            (network, "decoder", network.decoder),
            (network, "transformer_decoder", network.transformer_decoder),
            (network, "project_bottleneck_embed", network.project_bottleneck_embed),
            (network, "project_text_embed", network.project_text_embed),
            (network, "project_to_decoder_channels", network.project_to_decoder_channels),
        ]
    if target == "all":
        return [(None, None, network)]
    raise ValueError(f"Unknown LoRA target: {target}")


def apply_lora_to_linear_modules(
    network: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target: str,
    skip_multihead_attention: bool = True,
) -> int:
    replaced = 0
    for parent, root_name, root in _lora_target_specs(network, target):
        if target == "attention":
            continue
        if isinstance(root, nn.Linear):
            if parent is None or root_name is None:
                raise RuntimeError("Cannot replace the top-level network module with LoRA.")
            setattr(parent, root_name, LoRALinear(root, rank, alpha, dropout))
            replaced += 1
            continue
        for module in root.modules():
            if isinstance(module, (LoRALinear, LoRAMultiheadAttention)):
                continue
            if skip_multihead_attention and isinstance(module, nn.MultiheadAttention):
                continue
            for child_name, child in list(module.named_children()):
                if isinstance(child, (LoRALinear, LoRAMultiheadAttention)):
                    continue
                if skip_multihead_attention and isinstance(child, nn.MultiheadAttention):
                    continue
                if isinstance(child, nn.Linear):
                    setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
                    replaced += 1
    if replaced == 0 and target != "attention":
        raise RuntimeError(f"No nn.Linear modules found for LoRA target '{target}'.")
    return replaced


def apply_lora_to_attention_modules(
    network: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target: str,
    matrices: str,
) -> int:
    if matrices == "none":
        return 0
    replaced = 0
    for _, _, root in _lora_target_specs(network, target):
        for module in root.modules():
            if isinstance(module, LoRAMultiheadAttention):
                continue
            for child_name, child in list(module.named_children()):
                if isinstance(child, LoRAMultiheadAttention):
                    continue
                if isinstance(child, nn.MultiheadAttention):
                    setattr(
                        module,
                        child_name,
                        LoRAMultiheadAttention(
                            child,
                            rank=rank,
                            alpha=alpha,
                            dropout=dropout,
                            matrices=matrices,
                        ),
                    )
                    replaced += 1
    if replaced == 0:
        raise RuntimeError(f"No nn.MultiheadAttention modules found for LoRA target '{target}'.")
    return replaced


def configure_trainable_parameters(
    network: torch.nn.Module,
    trainable: str,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    lora_target: str = "prompt_decoder",
    lora_attention: str = "none",
) -> list[torch.nn.Parameter]:
    for param in network.parameters():
        param.requires_grad = False

    if trainable == "lora":
        linear_replaced = apply_lora_to_linear_modules(
            network,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target=lora_target,
        )
        attention_replaced = apply_lora_to_attention_modules(
            network,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target=lora_target,
            matrices=lora_attention,
        )
        if linear_replaced + attention_replaced == 0:
            raise RuntimeError("No modules were selected for LoRA.")
        for module in network.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True
            elif isinstance(module, LoRAMultiheadAttention):
                for name, param in module.named_parameters(recurse=False):
                    if name.startswith("lora_"):
                        param.requires_grad = True
    elif trainable == "all":
        modules = [network]
    elif trainable == "decoder":
        modules = [
            network.decoder,
            network.transformer_decoder,
            network.project_bottleneck_embed,
            network.project_text_embed,
            network.project_to_decoder_channels,
        ]
    elif trainable == "prompt_decoder":
        modules = [
            network.transformer_decoder,
            network.project_bottleneck_embed,
            network.project_text_embed,
            network.project_to_decoder_channels,
        ]
    else:
        raise ValueError(f"Unknown trainable mode: {trainable}")

    if trainable != "lora":
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True

    trainable_params = [p for p in network.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters selected.")
    return trainable_params


def expand_text_embeddings(text_embeddings: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    text_embeddings = text_embeddings.to(device)
    if text_embeddings.shape[0] == batch_size:
        return text_embeddings
    if text_embeddings.shape[0] != 1:
        raise ValueError(f"Cannot expand text embeddings with batch dim {text_embeddings.shape[0]} to {batch_size}")
    return text_embeddings.repeat(batch_size, 1, 1)
