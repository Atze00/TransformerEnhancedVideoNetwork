import torch.nn as nn
from x_transformers_mod import AttentionLayers
from x_transformers_mod import ContinuousTransformerWrapper
from movinets import MoViNet
from movinets.config import _C
from movinets.models import CausalModule

def _forward_impl(self, x):
    x = self.conv1(x)
    x = self.blocks(x)
    x = self.conv7(x)
    x = nn.AdaptiveAvgPool3d((x.shape[2],1,1))(x).squeeze(-1).squeeze(-1)
    return x

MoViNet._forward_impl = _forward_impl

MODELS = {
        "A0": _C.MODEL.MoViNetA0,
        "A1": _C.MODEL.MoViNetA1,
        "A2": _C.MODEL.MoViNetA2,
        "A3": _C.MODEL.MoViNetA3,
        "A4": _C.MODEL.MoViNetA4,
        "A5": _C.MODEL.MoViNetA5,
}

DIM_IN = {
        "A0": 480,
        "A1": 600,
        "A2": 640,
        "A3": 744,
        "A4": 856,
        "A5": 992,
}


class EnhancedVideoNetwork(CausalModule):
    def __init__(self, *, 
                 model_name,
                 causal,
                 pretrained,
                 dim_out,
                 max_seq_len,
                 max_mem_len,
                 dim_att,
                 depth,
                 heads,
                 rotary_pos_emb
                ):
        super().__init__()
        self.backbone = MoViNet( MODELS[model_name], causal = causal, pretrained = pretrained)
        dim_in = DIM_IN[model_name]
        self.transformer = ContinuousTransformerWrapper(
            dim_in = dim_in,
            dim_out = dim_out,
            max_seq_len = max_seq_len,
            max_mem_len = max_mem_len,
            attn_layers = AttentionLayers(
                causal = causal,
                dim = dim_att,
                depth = depth,
                heads = heads,
                rotary_pos_emb = rotary_pos_emb
            )
        )
    def forward(self, x):
        x = self.backbone(x).permute(0,2,1)
        x, mems = self.transformer(x, mems = self.activation, return_mems = True)
        self.activation = mems
        return x
    
    @staticmethod
    def _clean_activation_buffers(m):
        if issubclass(type(m), CausalModule):
            m.reset_activation()

    def clean_activation_buffers(self) -> None:
        self.apply(self._clean_activation_buffers)
