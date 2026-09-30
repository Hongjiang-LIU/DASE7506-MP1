"""SVD-initialized trainable low-rank projections, retaining FP32 inference."""
from torch import nn
from efficient import EfficientGPT
from refined import RefinedGPT


class LowRankLinear(nn.Module):
    def __init__(self,in_features,out_features,rank):
        super().__init__()
        self.left=nn.Linear(in_features,rank,bias=False)
        self.right=nn.Linear(rank,out_features,bias=False)
    def forward(self,x):
        return self.right(self.left(x))


class LowRankGPT(EfficientGPT):
    def __init__(self,config):
        super().__init__(config)
        rank=config['lowrank_rank']
        for block in self.blocks:
            for module,name in [(block,'qkv'),(block,'proj'),(block.mlp,'gate'),(block.mlp,'value'),(block.mlp,'out')]:
                original=getattr(module,name)
                setattr(module,name,LowRankLinear(original.in_features,original.out_features,rank))
    predict_log_probs=RefinedGPT.predict_log_probs


def build_model(config):
    return LowRankGPT(config)
