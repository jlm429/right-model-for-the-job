"""Aggregate figures and locally generated image galleries."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import CLASSES, dataset, metrics_from_logits, read

plt.rcParams.update({'font.size': 10, 'figure.facecolor': 'white'})


def learning_curves(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    for ax, metric in zip(axes, ['loss','accuracy','macro_f1']):
        for group in ['train','validation']:
            points = [(h['epoch'],h[group][metric]) for h in history if h.get(group) and metric in h[group]]
            if points:
                ax.plot(*zip(*points), 'o-', label=group)
        boundary = next((h['epoch']-.5 for h in history if h['stage']=='finetune'), None)
        if boundary:
            ax.axvline(boundary, color='gray', ls='--', label='unfreeze')
        ax.set(xlabel='Epoch', ylabel=metric.replace('_',' '))
        if ax.lines:
            ax.legend()
        ax.grid(alpha=.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def confusion_plot(metrics, directory):
    for name in ['confusion_raw','confusion_row_normalized']:
        values = np.asarray(metrics[name])
        fig, ax = plt.subplots(figsize=(8,7), constrained_layout=True)
        image = ax.imshow(values, cmap='Blues')
        fig.colorbar(image, ax=ax)
        ax.set(xticks=range(10), yticks=range(10), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel='Predicted class', ylabel='True class', title=name.replace('_',' ').title())
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        for i in range(10):
            for j in range(10):
                ax.text(j,i,f'{values[i,j]:.2f}' if 'normalized' in name else str(values[i,j]),
                        ha='center',va='center',fontsize=7,color='white' if values[i,j]>.6*values.max() else 'black')
        fig.savefig(Path(directory)/(name+'.png'), dpi=180)
        plt.close(fig)


def prediction_gallery(images, logits, labels, path):
    _, probabilities = metrics_from_logits(logits, labels)
    pred = logits.argmax(1)
    errors = np.flatnonzero(pred!=labels)
    indices = sorted(errors, key=lambda i: (-probabilities[i,pred[i]],int(i)))[:6]
    if not indices:
        return
    fig, axes = plt.subplots(2,3,figsize=(9,6),constrained_layout=True)
    for ax in axes.flat:
        ax.axis('off')
    for ax, i in zip(axes.flat, indices):
        ax.imshow(images[i])
        ax.set_title(f'#{i}: {CLASSES[labels[i]]} → {CLASSES[pred[i]]}\np={probabilities[i,pred[i]]:.3f} (uncalibrated)',fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def saved_gallery(metrics_dir, data_dir, output, download):
    entries = read(Path(metrics_dir)/'gallery.json')['errors']
    images = dataset(data_dir, False, download=download).data
    fig, axes = plt.subplots(2,3,figsize=(9,6),constrained_layout=True)
    for ax in axes.flat:
        ax.axis('off')
    for ax, entry in zip(axes.flat, entries):
        ax.imshow(images[entry['image_id']])
        ax.set_title(f"#{entry['image_id']}: {entry['true']} → {entry['predicted']}\np={entry['confidence']:.3f} (uncalibrated)", fontsize=9)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Recreate a historical error gallery locally, without inference.')
    parser.add_argument('--metrics-dir',type=Path,default=Path('results/resnet50'))
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    parser.add_argument('--output',type=Path,default=Path('runs/historical_errors.png'))
    parser.add_argument('--no-download',action='store_true')
    args=parser.parse_args()
    saved_gallery(args.metrics_dir,args.data_dir,args.output,not args.no_download)
