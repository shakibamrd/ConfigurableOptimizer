from __future__ import annotations

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from confopt.searchspace import NASBench201SearchSpace

supernet = NASBench201SearchSpace()
x = torch.randn(2, 3, 32, 32).to(device)

compiled_model = torch.compile(supernet)  # PyTorch 2.0 only

out, logits = supernet(x)

print(out.shape)
print(logits.shape)
print("Done.")
