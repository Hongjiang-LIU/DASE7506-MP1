import torch
from lowrank import build_model
from test_contract import ContractTests


class LowRankContractTests(ContractTests):
    def setUp(self):
        torch.set_num_threads(2)
        self.model=build_model(dict(vocab=2048,width=32,heads=4,depth=2,context=256,
            lowrank_rank=8,temperature=1.1,cache_weight=.05,cache_theta=15.)).eval()
