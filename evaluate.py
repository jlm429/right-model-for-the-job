"""Evaluate a selected checkpoint once and generate diagnostics from cached logits."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (dataset, device_for, make_model, metrics_from_logits, preprocessing, read,
                    save, seed_all, sha, synchronize)
from plots import confusion_plot, prediction_gallery


def evaluate(run, data_dir, device_name='auto', download=True):
    run = Path(run)
    config, selection = read(run/'config.json'), read(run/'selection.json')
    if sha(run/'best.pt') != selection['checkpoint_sha256'] or sha(run/'config.json') != selection['config_sha256']:
        raise ValueError('Selected checkpoint or configuration changed')
    cache = run/'test_predictions.npz'
    if not cache.exists():
        if (run/'test_started.json').exists():
            raise RuntimeError('A prior evaluation is incomplete. Reconcile it explicitly before any new test pass.')
        seed_all(config['seed'], config.get('threads', 4))
        device = device_for(device_name)
        model = make_model(config['experiment']).to(device)
        model.load_state_dict(torch.load(run/'best.pt', map_location='cpu', weights_only=True), strict=True)
        model.eval()
        test = dataset(data_dir, False, preprocessing(config['experiment']), download)
        loader = DataLoader(test, batch_size=config['batch_size'], num_workers=config.get('workers', config.get('num_workers', 0)),
                            generator=torch.Generator().manual_seed(config['seed']+2))
        logits_all, labels_all = [], []
        save(run/'test_started.json', dict(selection_sha256=sha(run/'selection.json')))
        synchronize(device)
        start = time.perf_counter()
        model_seconds = 0.
        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(device)
                synchronize(device)
                model_start = time.perf_counter()
                logits = model(images)
                synchronize(device)
                model_seconds += time.perf_counter()-model_start
                logits_all.append(logits.cpu().numpy())
                labels_all.append(labels.numpy())
        synchronize(device)
        elapsed = time.perf_counter()-start
        np.savez_compressed(cache, logits=np.concatenate(logits_all), labels=np.concatenate(labels_all), image_ids=np.arange(len(test)))
        save(run/'evaluation.json', dict(total_seconds=elapsed, model_seconds=model_seconds, device=str(device),
            images=len(test), test_forward_passes=1, cache_sha256=sha(cache),
            timing='Total: loader, preprocessing, transfer and model. Model: synchronized forward after input transfer.'))
    expected = read(run/'evaluation.json')['cache_sha256']
    if sha(cache) != expected:
        raise ValueError('Prediction cache changed')
    with np.load(cache) as predictions:
        logits, labels = predictions['logits'], predictions['labels']
        if not np.array_equal(predictions['image_ids'], np.arange(10000)):
            raise ValueError('Expected original complete CIFAR-10 test order')
    metrics, probabilities = metrics_from_logits(logits, labels)
    if metrics['images'] != 10000 or any(row['support']!=1000 for row in metrics['per_class']):
        raise ValueError('Expected complete balanced CIFAR-10 test set')
    save(run/'metrics.json', metrics)
    with (run/'per_class.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics['per_class'][0]))
        writer.writeheader()
        writer.writerows(metrics['per_class'])
    with (run/'predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['image_id','true_id','predicted_id','confidence'])
        for i, (label, pred) in enumerate(zip(labels, logits.argmax(1))):
            writer.writerow([i, int(label), int(pred), float(probabilities[i,pred])])
    confusion_plot(metrics, run)
    raw = dataset(data_dir, False, download=download)
    prediction_gallery(raw.data, logits, labels, run/'errors.png')
    print(f"Accuracy {metrics['accuracy']:.2%}; macro F1 {metrics['macro_f1']:.6f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--device', choices=['auto','mps','cuda','cpu'], default='auto')
    parser.add_argument('--no-download', action='store_true')
    args = parser.parse_args()
    evaluate(args.run, args.data_dir, args.device, not args.no_download)


if __name__ == '__main__':
    main()
