"""Offline behavioral checks; no dataset downloads, training runs, or API calls."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from common import make_model, metrics_from_logits, preprocessing, read, save, ROOT


class ExperimentsTest(unittest.TestCase):
    def test_architecture_shapes_and_parameter_counts(self):
        torch.set_num_threads(2)
        for name, count, size in [('original',356810,32), ('improved',357258,32), ('resnet50',23528522,224)]:
            model = make_model(name).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()), count)
            with torch.inference_mode():
                self.assertEqual(tuple(model(torch.zeros(1,3,size,size)).shape), (1,10))

    def test_deterministic_preprocessing(self):
        image = Image.fromarray(np.random.default_rng(42).integers(0,256,(32,32,3),dtype=np.uint8))
        for name, size in [('original',32),('improved',32),('resnet50',224)]:
            transform = preprocessing(name)
            self.assertTrue(torch.equal(transform(image), transform(image)))
            self.assertEqual(tuple(transform(image).shape), (3,size,size))

    def test_metrics_known_confusion_and_empty_class(self):
        labels = np.array([0,0,1,1])
        logits = np.full((4,10), -10.)
        logits[np.arange(4), [0,1,1,1]] = 10.
        metrics, probabilities = metrics_from_logits(logits, labels)
        self.assertEqual(metrics['accuracy'], .75)
        self.assertAlmostEqual(metrics['per_class'][0]['f1'], 2/3)
        self.assertAlmostEqual(metrics['per_class'][1]['f1'], .8)
        self.assertAlmostEqual(metrics['macro_f1'], (2/3+.8)/10)
        self.assertEqual(metrics['per_class'][9]['support'], 0)
        self.assertTrue(np.allclose(probabilities.sum(1), 1))
        with self.assertRaises(ValueError):
            metrics_from_logits(np.zeros((0,10)), [])

    def test_checkpoint_round_trip(self):
        model = make_model('improved').eval()
        images = torch.randn(2,3,32,32)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'weights.pt'
            torch.save(model.state_dict(),path)
            restored=make_model('improved').eval()
            restored.load_state_dict(torch.load(path,weights_only=True),strict=True)
            with torch.inference_mode():
                self.assertTrue(torch.equal(model(images),restored(images)))

    def test_training_refuses_existing_output(self):
        from train import train
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            save(path/'keep.json', {'existing': True})
            with self.assertRaises(FileExistsError):
                train(ROOT/'configs/original.json',path,path/'missing-data',download=False)
            self.assertEqual(read(path/'keep.json'), {'existing': True})

    def test_evaluation_rejects_changed_selection(self):
        from evaluate import evaluate
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            save(path/'config.json', {'experiment': 'original'})
            save(path/'selection.json', {'checkpoint_sha256': 'incorrect', 'config_sha256': 'incorrect'})
            (path/'best.pt').write_bytes(b'not a trusted checkpoint')
            with self.assertRaises(ValueError):
                evaluate(path,path/'missing-data',download=False)

    def test_saved_confusion_matches_reported_accuracy(self):
        for name in ['original','improved','resnet50']:
            m=read(ROOT/'results'/name/'metrics.json')
            cm=np.array(m['confusion_raw'])
            self.assertEqual(int(cm.sum()),10000)
            self.assertTrue(np.array_equal(cm.sum(1),np.full(10,1000)))
            self.assertAlmostEqual(cm.trace()/cm.sum(),m['accuracy'])


if __name__=='__main__':
    unittest.main()
