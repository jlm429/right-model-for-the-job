"""Shared CIFAR-10 models, deterministic split checks, and metrics."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights, resnet50

ROOT = Path(__file__).resolve().parent
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
WEIGHTS = ResNet50_Weights.IMAGENET1K_V2


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def device_for(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    return torch.device(name)


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def seed_all(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(experiment, pretrained=False):
    if experiment == 'resnet50':
        # Training always requests pretrained=True; evaluation loads its own checkpoint.
        model = resnet50(weights=WEIGHTS if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model
    if experiment not in {'original', 'improved'}:
        raise ValueError(f'Unknown experiment: {experiment}')
    layers = []
    incoming = 3
    for channels in (32, 64, 128):
        layers.append(nn.Conv2d(incoming, channels, 3, padding=1))
        if experiment == 'improved':
            layers.append(nn.BatchNorm2d(channels))
        layers.extend([nn.ReLU(), nn.MaxPool2d(2)])
        incoming = channels
    layers.append(nn.Flatten())
    if experiment == 'improved':
        layers.append(nn.Dropout(.3))
    layers.extend([nn.Linear(2048, 128), nn.ReLU()])
    if experiment == 'improved':
        layers.append(nn.Dropout(.3))
    layers.append(nn.Linear(128, 10))
    return nn.Sequential(*layers)


def preprocessing(experiment, training=False, config=None):
    config = config or {}
    if experiment == 'resnet50':
        if not training:
            return WEIGHTS.transforms()
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=tuple(config.get('crop_scale', [.8, 1.])),
                ratio=tuple(config.get('crop_ratio', [.9, 1.1])),
                interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize(WEIGHTS.transforms().mean, WEIGHTS.transforms().std)])
    operations = []
    if training and config.get('augmentation'):
        operations = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    return transforms.Compose(operations + [transforms.ToTensor(), transforms.Normalize((.5,)*3, (.5,)*3)])


def dataset(data_dir, training, transform=None, download=True):
    return datasets.CIFAR10(data_dir, train=training, transform=transform, download=download)


def split_indices(labels, path=ROOT/'splits/seed42.npz'):
    identity = read(ROOT/'splits/identity.json')
    if sha(path) != identity['split_file_sha256']:
        raise ValueError('Saved split checksum differs from the historical experiment')
    labels = np.asarray(labels, dtype=np.int64)
    if hashlib.sha256(labels.tobytes()).hexdigest() != identity['training_labels_sha256']:
        raise ValueError('Dataset label order differs from the historical experiment')
    with np.load(path) as archive:
        split = {name: archive[name].copy() for name in ['train', 'validation']}
    for name, per_class in [('train', 4500), ('validation', 500)]:
        indices = split[name]
        if hashlib.sha256(indices.tobytes()).hexdigest() != identity[name+'_indices_sha256']:
            raise ValueError('Historical split indices changed')
        if not np.array_equal(np.bincount(labels[indices], minlength=10), [per_class]*10):
            raise ValueError('Split class counts differ')
    if len(np.intersect1d(*split.values())) or not np.array_equal(np.union1d(*split.values()), np.arange(50000)):
        raise ValueError('Split overlaps or omits training images')
    return split


def metrics_from_logits(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.shape != (len(labels), 10) or not len(labels) or not np.isfinite(logits).all():
        raise ValueError('Expected finite N by 10 logits and nonempty labels')
    if (labels < 0).any() or (labels > 9).any():
        raise ValueError('Labels must be CIFAR-10 class indices')
    predictions = logits.argmax(1)
    logp = logits - np.logaddexp.reduce(logits, axis=1)[:, None]
    cm = np.zeros((10, 10), dtype=np.int64)
    np.add.at(cm, (labels, predictions), 1)
    support = cm.sum(1)
    precision = np.divide(cm.diagonal(), cm.sum(0), out=np.zeros(10), where=cm.sum(0)>0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(10), where=support>0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(10), where=precision+recall>0)
    normalized = np.divide(cm, support[:, None], out=np.zeros((10, 10)), where=support[:, None]>0)
    pairs = sorted([(int(cm[i,j]), i, j) for i in range(10) for j in range(10) if i!=j],
                   key=lambda row: (-row[0], row[1], row[2]))
    result = dict(accuracy=float((labels==predictions).mean()), loss=float(-logp[np.arange(len(labels)), labels].mean()),
        correct=int((labels==predictions).sum()), images=len(labels), macro_f1=float(f1.mean()),
        weighted_f1=float(np.average(f1, weights=support)), classes=CLASSES,
        per_class=[dict(class_id=i, **{'class': CLASSES[i]}, precision=float(precision[i]),
                       recall=float(recall[i]), f1=float(f1[i]), support=int(support[i])) for i in range(10)],
        confusion_raw=cm.tolist(), confusion_row_normalized=normalized.tolist(),
        misclassification_pairs=[dict(true=CLASSES[i], predicted=CLASSES[j], count=n) for n,i,j in pairs if n],
        confidence='Uncalibrated softmax; no calibration fitted')
    return result, np.exp(logp)
