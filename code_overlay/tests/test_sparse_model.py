import torch
from sparse_ngram import build_model
from test_contract import ContractTests


class SparseContractTests(ContractTests):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.model=build_model(dict(vocab=2048,width=32,heads=4,depth=2,context=256,
            ngram_dim=8,ngram_buckets=[31,127],temperature=1.1,cache_weight=.05,cache_theta=15.)).eval()
