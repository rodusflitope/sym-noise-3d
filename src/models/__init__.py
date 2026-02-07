from .simple_eps import EpsilonMLP
from .pointnet_eps import PointNetEpsilon
from .pointtransformer_eps import PointTransformerEpsilon
from .latent_eps import LatentEpsilonMLP
from .autoencoder import PointAutoencoder
from .lion_ae import LionAutoencoder
from .lion_ae_mlp import LionAutoencoder as LionAutoencoderMLP
from .builder import build_model

try:
    from .lion_ae_legacy import LionAutoencoder as LionAutoencoderLegacy
except Exception:
    LionAutoencoderLegacy = None

try:
    from .lion_two_priors import LionTwoPriorsDDM
except Exception:
    LionTwoPriorsDDM = None

__all__ = [
    "EpsilonMLP",
    "PointNetEpsilon",
    "PointTransformerEpsilon",
    "LatentEpsilonMLP",
    "PointAutoencoder",
    "LionAutoencoder",
    "LionAutoencoderMLP",
    "LionAutoencoderLegacy",
    "build_model",
    "LionTwoPriorsDDM",
]
