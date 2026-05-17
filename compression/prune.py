import torch
import torch.nn.utils.prune as prune
import torch_pruning as tp

def unstructured_pruning(model, config):
    sparsity = config.get('sparsity', 0.3)
    prune_conv2d = config.get('prune_conv2d', True)
    prune_linear = config.get('prune_linear', False)
    exclude_phase = config.get('exclude_phase_decoder', False)
    
    prune_fn = prune.l1_unstructured # TODO: не хардкодить это

    for name, module in model.named_modules():
        # Если exclude_phase=True и слой из phase_decoder, то не применяем к нему прунинг 
        if exclude_phase and 'phase_decoder' in name:
            print(f"Skipping phase decoder module: {name}")
            continue

        # Conv2d слои
        if prune_conv2d and isinstance(module, torch.nn.Conv2d):
            prune_fn(module, name='weight', amount=sparsity)
            prune.remove(module, 'weight')
            print(f"Pruned Conv2d: {name}")

        # Linear слои    
        elif prune_linear and isinstance(module, torch.nn.Linear):
            prune_fn(module, name='weight', amount=sparsity)
            prune.remove(module, 'weight')
            print(f"Pruned Linear: {name}")

        # MultiheadAttention слои
        elif prune_linear and isinstance(module, torch.nn.MultiheadAttention):
            if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                prune_fn(module, name='in_proj_weight', amount=sparsity)
                prune.remove(module, 'in_proj_weight')
                print(f"Pruned MultiheadAttention.in_proj_weight: {name}")
            if hasattr(module, 'out_proj') and hasattr(module.out_proj, 'weight'):
                prune_fn(module.out_proj, name='weight', amount=sparsity)
                prune.remove(module.out_proj, 'weight')
                print(f"Pruned MultiheadAttention.out_proj.weight: {name}")

    global_sparsity, total_params, zero_params = compute_global_sparsity(model)
    print(f"Global sparsity after pruning: {global_sparsity}")
    print(f"Total params: {total_params}, zero params: {zero_params}")
    
    return model


def compute_global_sparsity(model):
    """
    Подсчитывает общее количество нулевых весов и общее количество параметров во всей модели.
    Возвращает:
        global_sparsity (float): доля нулевых весов среди всех параметров.
        total_params (int): общее количество параметров.
        zero_params (int): количество нулевых параметров.
    """
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        zeros = (param.data == 0).sum().item()
        total = param.numel()
        total_params += total
        zero_params += zeros
    global_sparsity = zero_params / total_params if total_params > 0 else 0.0
    return global_sparsity, total_params, zero_params