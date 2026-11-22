import pytest


def pytest_collection_modifyitems(config, items):
    gpu_items = [item for item in items if "gpu" in item.keywords]
    if not gpu_items:
        return
    import torch

    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="needs CUDA")
        for item in gpu_items:
            item.add_marker(skip)
