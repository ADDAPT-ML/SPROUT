"""Optional bitwise-reproducible execution (`--deterministic`).

The paper's runs used the default (non-deterministic) CUDA kernels. Deterministic mode is used to check that
two implementations produce identical outputs; it must be configured before torch is imported.
"""
import os
import random
import sys


def configure(seed, mask2former=False, argv=None):
    """Re-execute the interpreter (with `argv`, default sys.argv) under a fixed hash seed and cuBLAS workspace
    if needed, then seed every generator and force deterministic kernels. mask2former=True also patches the
    GenEval detector (see below)."""
    required = {'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONHASHSEED': str(seed)}
    if any(os.environ.get(k) != v for k, v in required.items()):
        os.execve(sys.executable, [sys.executable, *(argv or sys.argv)], dict(os.environ, **required))
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if mask2former:
        _patch_mask2former()


def _patch_mask2former():
    """MMDetection 2.28.2 clears fully masked attention rows with an advanced-index assignment, whose
    deterministic CUDA kernel in torch 1.12 mis-broadcasts the scalar (pytorch/pytorch#79987). Replace it with
    the equivalent masked_fill_. Only used in deterministic mode."""
    import inspect
    import re
    import textwrap
    import mmdet
    import torch
    from mmdet.models.dense_heads.mask2former_head import Mask2FormerHead
    if mmdet.__version__ != '2.28.2' or not torch.__version__.startswith('1.12.'):
        raise RuntimeError('The Mask2Former patch is written for mmdet 2.28.2 and torch 1.12')
    original = Mask2FormerHead.forward
    source = textwrap.dedent(inspect.getsource(original))
    pattern = r'attn_mask\[torch\.where\(\s*attn_mask\.sum\(-1\) == attn_mask\.shape\[-1\]\)\] = False'
    replacement = 'attn_mask.masked_fill_((attn_mask.sum(-1) == attn_mask.shape[-1]).unsqueeze(-1), False)'
    patched, count = re.subn(pattern, replacement, source)
    if count != 1:
        raise RuntimeError('Expected exactly one attention-mask assignment in Mask2FormerHead.forward')
    namespace = {}
    exec(compile(patched, inspect.getsourcefile(original), 'exec'), original.__globals__, namespace)
    Mask2FormerHead.forward = namespace['forward']
