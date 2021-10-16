# malay-VITS

Originally from https://github.com/jaywalnut310/vits

## how-to train

1. Build Monotonic Alignment Search,

```bash
cd monotonic_align
python3 setup.py build_ext --inplace
```

2. Prepare dataset,

- For `male` speaker, [prepare-male-vits.ipynb](prepare-male-vits.ipynb).
- For `female` speaker, [prepare-female-vits.ipynb](prepare-female-vits.ipynb).
- For `haqkiem` speaker, [prepare-haqkiem-vits.ipynb](prepare-haqkiem-vits.ipynb).

## Citation

```bibtex
@misc{kim2021conditional,
      title={Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech}, 
      author={Jaehyeon Kim and Jungil Kong and Juhee Son},
      year={2021},
      eprint={2106.06103},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```