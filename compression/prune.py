import torch
import torch.nn.utils.prune as prune

def unstructured_pruning(model, config):
    sparsity = config.get('sparsity', 0.3)
    prune_conv2d = config.get('prune_conv2d', True)
    prune_linear = config.get('prune_linear', False)
    exclude_phase = config.get('exclude_phase_decoder', False)
    
    prune_fn = prune.l1_unstructured

    for name, module in model.named_modules():
        if exclude_phase and 'phase_decoder' in name:
            print(f"Skipping phase decoder module: {name}")
            continue
            
        if prune_conv2d and isinstance(module, torch.nn.Conv2d):
            prune_fn(module, name='weight', amount=sparsity)
            prune.remove(module, 'weight')
            print(f"Pruned Conv2d: {name}")
        elif prune_linear and isinstance(module, torch.nn.Linear):
            prune_fn(module, name='weight', amount=sparsity)
            prune.remove(module, 'weight')
            print(f"Pruned Linear: {name}")
    
    return model