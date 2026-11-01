"""Model architectures and model construction."""

from .dehazeformer import DehazeFormer, dehazeformer_t
from .dea_net import DeaNet, dea_net
from .recent_lightweight import SLRNet, WLDNet, slrnet, wldnet


def build_model(name: str, *, use_depth: bool = False, base_dim: int = 32):
    """Construct an architecture by its experiment-config name."""
    normalized = name.lower().replace("-", "_")
    # MCBM is an online training augmentation; it does not alter DehazeFormer-T.
    variants = {
        "dehazeformer_t_m1": {"fusion": "scaf"},
        "dehazeformer_t_m2": {"fusion": "scaf", "haze_guidance": True},
        "dehazeformer_t_m3": {
            "fusion": "scaf", "haze_guidance": True, "frequency_branch": True,
        },
        "dehazeformer_t_m4": {"wavelet_refinement": True},
        "dehazeformer_t_m5": {
            "fusion": "scaf", "haze_guidance": True, "haze_skip_guidance": True,
        },
        "dehazeformer_t_m6": {"physical_consistency": True},
        "dehazeformer_t_m7": {"decoder_skip_modulation": True},
        "dehazeformer_t_m8": {"dcp_position_embedding": True},
        "dehazeformer_t_m9": {
            "decoder_skip_modulation": True, "dcp_position_embedding": True,
        },
        "m1": {"fusion": "scaf"},
        "m2": {"fusion": "scaf", "haze_guidance": True},
        "m3": {"fusion": "scaf", "haze_guidance": True, "frequency_branch": True},
        "m4": {"wavelet_refinement": True},
        "m5": {"fusion": "scaf", "haze_guidance": True, "haze_skip_guidance": True},
        "m6": {"physical_consistency": True},
        "m7": {"decoder_skip_modulation": True},
        "m8": {"dcp_position_embedding": True},
        "m9": {"decoder_skip_modulation": True, "dcp_position_embedding": True},
    }
    if normalized in variants:
        if use_depth:
            raise ValueError("M1-M9 ablation variants are RGB-only experiments.")
        return dehazeformer_t(**variants[normalized])
    if normalized in {"dehazeformer_t", "dehazeformer", "dehazeformer_t_mcbm"}:
        return dehazeformer_t(use_depth=use_depth)
    if normalized in {"dea_net", "deanet"}:
        if use_depth:
            raise ValueError("DEA-Net experiment currently supports RGB input only.")
        return dea_net(base_dim=base_dim)
    if normalized in {"slrnet", "slr_net"}:
        if use_depth:
            raise ValueError("SLRNet experiment currently supports RGB input only.")
        return slrnet(base_channels=base_dim)
    if normalized in {"wldnet", "wld_net", "wavelite_dehaze"}:
        if use_depth:
            raise ValueError("WLD-Net experiment currently supports RGB input only.")
        return wldnet(channels=base_dim)
    raise ValueError(f"Unsupported model architecture: {name!r}")


__all__ = [
    "DehazeFormer", "dehazeformer_t", "DeaNet", "dea_net", "SLRNet", "WLDNet",
    "slrnet", "wldnet", "build_model",
]
