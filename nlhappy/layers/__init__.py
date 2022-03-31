from .normalization import LayerNorm
from .loss import MultiLabelCategoricalCrossEntropy
from .classifier import GlobalPointer, EfficientGlobalPointer, CRF, SimpleDense
from .embedding import SinusoidalPositionEmbedding
from .activation import GELU, SWISH, GELU_Approximate

activations = {'gelu':GELU, 'swish':SWISH,'gelu_approximate':GELU_Approximate}
