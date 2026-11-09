"""Train a named historical configuration; test evaluation is a separate command."""
import argparse
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset

from common import (ROOT, dataset, device_for, make_model, metrics_from_logits, preprocessing,
                    read, save, seed_all, sha, split_indices, synchronize)


def run_epoch(model, loader, device, optimizer=None, frozen=False, max_batches=None):
    model.train(optimizer is not None and not frozen)
    if optimizer is not None and frozen:
        model.fc.train()
    losses = 0.
    logits_all, labels_all = [], []
    with torch.set_grad_enabled(optimizer is not None):
        for index, (images, labels) in enumerate(loader):
            images, target = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            losses += loss.item()*len(labels)
            logits_all.append(logits.detach().cpu().numpy())
            labels_all.append(labels.numpy())
            if max_batches and index+1 >= max_batches:
                break
    metrics, _ = metrics_from_logits(np.concatenate(logits_all), np.concatenate(labels_all))
    return {key: metrics[key] for key in ['accuracy', 'macro_f1', 'images']} | {'loss': losses/metrics['images']}


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train(config_path, output, data_dir, device_name='auto', download=True, max_batches=None):
    config = read(config_path)
    name = config['experiment']
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty run directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    config['smoke_max_batches'] = max_batches
    config['comparable_to_historical'] = max_batches is None
    seed_all(config['seed'], config.get('threads', 4))
    device = device_for(device_name)
    save(output/'config.json', config)
    save(output/'environment.json', dict(python=platform.python_version(), torch=torch.__version__,
        torchvision=torchvision.__version__, numpy=np.__version__, device=str(device), threads=torch.get_num_threads(),
        note='Fixed seeds; device kernels and package versions may prevent bitwise reproduction.'))
    training = dataset(data_dir, True, preprocessing(name, True, config), download)
    if name == 'original':
        train_data, validation_data = training, None
    else:
        split = split_indices(training.targets)
        deterministic = dataset(data_dir, True, preprocessing(name, False, config), False)
        train_data = Subset(training, split['train'])
        validation_data = Subset(deterministic, split['validation'])
    # Original baseline used the global RNG for shuffling; later studies used a dedicated generator.
    generator = None if name == 'original' else torch.Generator().manual_seed(config['seed'])
    workers = config.get('workers', config.get('num_workers', 0))
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True,
                              num_workers=workers, generator=generator)
    val_loader = None if validation_data is None else DataLoader(validation_data, batch_size=config['batch_size'],
        num_workers=workers, generator=torch.Generator().manual_seed(config['seed']+1))
    model = make_model(name, pretrained=name=='resnet50').to(device)
    resnet = name=='resnet50'
    stages = [('head', config['head_epochs']), ('finetune', config['finetune_epochs'])] if resnet else [('cnn', config['max_epochs'])]
    best = -1.
    history = []
    synchronize(device)
    started = time.perf_counter()
    # The recorded ResNet run had a one-hour wall budget including a five-minute evaluation reserve.
    budget = config.get('budget_seconds')
    reserve = config.get('reserve_seconds', 0)
    deadline_reached = False
    for stage, epochs in stages:
        if resnet:
            if stage=='finetune':
                model.load_state_dict(torch.load(output/'head_best.pt', map_location='cpu', weights_only=True))
            for parameter_name, parameter in model.named_parameters():
                parameter.requires_grad_(stage=='finetune' or parameter_name.startswith('fc.'))
            groups = [{'params': list(model.fc.parameters()), 'lr': config['head_lr']}]
            if stage=='finetune':
                groups = [dict(params=[p for n,p in model.named_parameters() if not n.startswith('fc.')], lr=config['backbone_lr']),
                          dict(params=list(model.fc.parameters()), lr=config['finetune_head_lr'])]
            optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=tuple(config['betas']),
                eps=config['eps'], weight_decay=config['weight_decay'], amsgrad=config['amsgrad'])
            scheduler = None
        stage_best, stale = -1., 0
        for epoch in range(1, epochs+1):
            elapsed = time.perf_counter()-started
            observed = [row['epoch_seconds'] for row in history if row['stage']==stage]
            if budget and elapsed + reserve + (max(observed)*1.2 if observed else 0) >= budget:
                deadline_reached = True
                break
            synchronize(device)
            epoch_start = time.perf_counter()
            rates = [group['lr'] for group in optimizer.param_groups]
            training_metrics = run_epoch(model, train_loader, device, optimizer, frozen=stage=='head', max_batches=max_batches)
            validation = run_epoch(model, val_loader, device, max_batches=max_batches) if val_loader else None
            # The historical baseline selected its fixed final epoch, without validation.
            score = validation['accuracy'] if validation else float(epoch)
            if score > best:
                best = score
                torch.save(cpu_state(model), output/'best.pt')
            if score > stage_best:
                stage_best, stale = score, 0
                if stage=='head':
                    torch.save(cpu_state(model), output/'head_best.pt')
            else:
                stale += 1
            if scheduler:
                scheduler.step()
            synchronize(device)
            row = dict(epoch=len(history)+1, stage=stage, stage_epoch=epoch, learning_rates=rates,
                train=training_metrics, validation=validation, epoch_seconds=time.perf_counter()-epoch_start)
            history.append(row)
            save(output/'history.json', history)
            print(f"{stage} {epoch}/{epochs}: train={training_metrics['accuracy']:.4f}; validation={validation}", flush=True)
            if config.get('patience') and stale >= config['patience']:
                break
        if deadline_reached:
            break
    if not history:
        raise RuntimeError('Budget allowed no complete epoch')
    selected = next(row for row in history if (row['validation']['accuracy'] if row['validation'] else row['epoch'])==best)
    synchronize(device)
    save(output/'selection.json', dict(selected_epoch=selected['epoch'], selected_stage=selected['stage'],
        best_validation_accuracy=selected['validation']['accuracy'] if selected['validation'] else None,
        selection_rule='Highest validation accuracy, earliest epoch on ties' if val_loader else 'Fixed final epoch; no validation',
        parameters=sum(p.numel() for p in model.parameters()), training_seconds=time.perf_counter()-started,
        checkpoint_sha256=sha(output/'best.pt'), config_sha256=sha(output/'config.json'),
        split_sha256=sha(ROOT/'splits/seed42.npz') if val_loader else None,
        locked_at=datetime.now(timezone.utc).isoformat(), test_evaluated=False, budget_stopped=deadline_reached))
    from plots import learning_curves
    learning_curves(history, output/'learning.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--device', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--no-download', action='store_true')
    parser.add_argument('--smoke-max-batches', type=int, help='Limit batches per epoch; results are explicitly non-comparable')
    args = parser.parse_args()
    if args.smoke_max_batches is not None and args.smoke_max_batches < 1:
        parser.error('--smoke-max-batches must be positive')
    train(args.config, args.output, args.data_dir, args.device, not args.no_download, args.smoke_max_batches)


if __name__ == '__main__':
    main()
