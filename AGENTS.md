# Project guidelines

This repository is a reproducible CIFAR-10 case study about choosing an appropriate model for a task.
Keep the implementation straightforward, portable, and focused on the experiments.

## Security and publication

- Never read, print, hardcode, commit, or push API keys, tokens, credentials, `.env` files, or service account files.
- Retrieve notebook API keys from Google Colab Secrets using the variable name `OPENAI_API_KEY`.
- Keep all paid API execution disabled by default and require explicit user-controlled execution and limits.
- Exclude downloaded datasets, model checkpoints, caches, virtual environments, machine-specific paths, and private operational files from Git.
- Inspect the staged file list before committing and the outgoing changes before pushing.
- Do not run paid inference or full training during routine validation.

## Preserve experimental evidence

- Preserve historical source experiments, datasets, split indices, checkpoints, and results.
- Write new experiment outputs to a distinct run directory and refuse accidental overwrites.
- Preserve the recorded architectures, preprocessing, random seeds, optimizers, and hyperparameters when reproducing a named experiment.
- Label changes to a historical configuration as a new experiment.
- Reuse the saved seed-42 stratified 45,000/5,000 training-validation split for the improved CNN and ResNet-50.
- The original five-epoch CNN used all 50,000 training examples and had no validation selection.
- Apply augmentation only to training data; validation and test transforms must be deterministic.
- Select checkpoints and hyperparameters using validation data and keep test evaluation separate from training.
- Record configuration, per-epoch metrics, model selection, device, versions, timing scope, and results.
- Support MPS, CUDA, and CPU without hardcoded local paths.
- ResNet-50 is ImageNet pretrained transfer learning, never training from scratch under that experiment name.

## Reporting and fairness

- Never fabricate metrics, learning curves, predictions, API usage, latency, costs, or completed experiments.
- Distinguish locally verified artifacts, externally supplied results, and hypotheses in prose and machine-readable summaries.
- Keep the local 10,000-image test results separate from the externally supplied 100-image balanced validation pilot.
- Do not present the 94.03% ResNet-50 and 83% frontier pilot as a controlled head-to-head comparison.
- Treat ten-image smoke tests as exploratory and do not rank models from them.
- Explain supervised task-specific training, ImageNet pretraining, and zero-shot API inference clearly.
- Discuss frontier models' strengths and limitations with a balanced technical tone.
- Check redistribution rights and attribution before publishing image examples; the code license does not automatically license datasets or pretrained weights.
- Document ViT, parameter-efficient adaptation, paired frontier evaluation, and hybrid inference as future work until measured.

## Engineering and validation

- Keep changes small, direct, and reviewable; avoid unnecessary frameworks or external services.
- Provide separate training and evaluation commands, clear dependencies, and reproduction instructions.
- Exercise imports, CLI help, model construction, preprocessing, checkpoint compatibility, metrics, and documented commands with lightweight checks.
- Test observable behavior rather than matching strings in source files.
- Never overwrite unrelated work or add an agent as a commit co-author.
- Do not use em dashes in prose, comments, documentation, or commit messages.
- Add or change a license only when the owner has chosen one.
