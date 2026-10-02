#!/usr/bin/env python3
"""Export exact librosa band membership for Deux; never recompute it at runtime."""
import json
import sys
from pathlib import Path
import librosa
import numpy as np

bank = librosa.filters.mel(sr=44100, n_fft=2048, n_mels=60)
bank[0, 0] = bank[-1, -1] = 1
bands = [np.flatnonzero(row > 0).tolist() for row in bank]
config = dict(family="mel-band-roformer", dim=256, depth=12, heads=8,
              head_dim=64, ff_mult=4, mask_depth=2, mask_expansion=4,
              sample_rate=44100, n_fft=2048, hop=441, chunk_samples=573300,
              overlap=2, zero_dc=True, stems=["vocals", "instrumental"], bands=bands)
destination = Path(sys.argv[1])
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(config, indent=2) + "\n")
