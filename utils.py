import numpy as np


def get_n_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    print("-"*80)
    print("Number of trainable parameters:", str(n_params))
    print("-"*80)