# malay-VITS

Originally from https://github.com/jaywalnut310/vits

## how-to train

1. Build Monotonic Alignment Search,

```bash
cd monotonic_align
python3 setup.py build_ext --inplace
```

2. Prepare dataset,

- For `osman` speaker, [prepare-osman-vits.ipynb](prepare-osman-vits.ipynb).
- For `yasmin` speaker, [prepare-yasmin-vits.ipynb](prepare-yasmin-vits.ipynb).

3. Run training,

- For `osman` speaker,

```bash
python3 train.py -c config/osman.json -m osman
```

- For `yasmin` speaker,

```bash
python3 train.py -c config/yasmin.json -m yasmin
```

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