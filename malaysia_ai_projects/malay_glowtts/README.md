# malay-GlowTTS

Originally from https://github.com/jaywalnut310/glowtts

## how-to train

1. Build Monotonic Alignment Search,

```bash
cd monotonic_align
python3 setup.py build_ext --inplace
```

2. Prepare dataset,

- For `male` speaker, [prepare-male-glowtts.ipynb](prepare-male-glowtts.ipynb).
- For `female` speaker, [prepare-female-glowtts.ipynb](prepare-female-glowtts.ipynb).
- For `haqkiem` speaker, [prepare-haqkiem-glowtts.ipynb](prepare-haqkiem-glowtts.ipynb).

## Citation

```bibtex
@misc{kim2020glowtts,
      title={Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search}, 
      author={Jaehyeon Kim and Sungwon Kim and Jungil Kong and Sungroh Yoon},
      year={2020},
      eprint={2005.11129},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```