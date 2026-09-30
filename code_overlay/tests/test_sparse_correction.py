import torch
from fast_refined import FastRefinedGPT
from sparse_correction import SparseCorrectionGPT,corrected_targets
from test_ngrams import small_tables
from test_contract import ContractTests


class CorrectionContractTests(ContractTests):
    def setUp(self):
        torch.set_num_threads(2)
        buffers=small_tables()
        config=dict(vocab=2048,width=32,heads=4,depth=2,context=256,
                    temperature=1.1,cache_weight=.05,cache_theta=15.,
                    correction_min=3,correction_max=5,correction_strength=.1,correction_scale=1.,
                    ngram_shapes={k:list(v.shape) for k,v in buffers.items()})
        self.model=SparseCorrectionGPT(config).eval()
        self.model.ngram.load_state_dict(buffers)

    def test_target_shortcut_matches_full_inference(self):
        base=FastRefinedGPT(self.model.config).eval()
        base.load_state_dict({k:v for k,v in self.model.state_dict().items() if not k.startswith('ngram.')})
        x=torch.tensor([[7,8,9,10,7],[7,8,11,10,8]])
        y=torch.tensor([[8,9,10,7,8],[8,11,10,8,7]])
        neural=base.predict_log_probs(x).exp().gather(-1,y[...,None]).squeeze(-1).numpy()
        target=corrected_targets(self.model.ngram.numpy_tables(),x.numpy(),y.numpy(),neural,3,5,.1,1.)
        actual=self.model.predict_log_probs(x).exp().gather(-1,y[...,None]).squeeze(-1)
        torch.testing.assert_close(actual,torch.from_numpy(target),atol=1e-7,rtol=1e-6)
