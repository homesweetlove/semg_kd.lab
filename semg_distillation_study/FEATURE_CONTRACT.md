# Trained teacher/student input contract

Latest robustness bundle paths: `portable_models/FAMILY/{teacher,direct,selected}/best.pt` and `portable_models/examples.npz`, inside the `semg_distillation_study` folder. The three `selected` students use the newly selected weights; read `portable_models/manifest.json`. Run `python -B verify_portable_models.py` from that project folder to check all nine models against 64 real source inputs. `teacher` and `direct` are preserved earlier full30 models; `selected` contains the three new full30 students. None has an independent test score. The mathematical input contract below is unchanged. The `kd` paths and root-layout example later in this document describe the previous bundle and remain historical.

```python
import numpy as np
from load_distilled import ElbowAngleModel
with np.load('portable_models/examples.npz', allow_pickle=False) as examples:
    model = ElbowAngleModel('portable_models/gru/selected/best.pt')
    angles = model.predict_degrees(examples['X'])
    np.testing.assert_allclose(angles, examples['gru__selected'], atol=5e-4, rtol=0)
```

The new nine-model bundle uses `load_distilled.ElbowAngleModel` with `FAMILY/{teacher,direct,kd}/best.pt` and the exact supplied `models.py`. The saved role reconstructs the large teacher or small student. Both consume extracted sEMG features, not raw voltage samples. The loader applies the checkpoint's normalization and converts its scalar output back to elbow-angle degrees. The older six-teacher bundle instead uses its own `load_teacher.TrainedTeacher` and `FAMILY/best.pt`; follow that archive's README for its legacy paths.

## Array and time order

- Input: finite float32-compatible array `[N, 21, 12]`.
- Time dimension: oldest to newest, `[t−2.0, t−1.9, …, t]` seconds, exactly100ms spacing.
- Every frame must come from the same person and same recording. The training pipeline rejects missing frames; it does not interpolate across missing rows or recording boundaries.
- Each frame uses only the current12 columns of the historical72-column feature array. Do not pass its six lag blocks as the time dimension.
- Output: one current elbow angle in degrees per sequence. Neither previous angle labels nor IMU values are predictors.

## Feature columns, in exact order

The channel order within every group is: biceps short head, biceps long head, triceps long head, triceps lateral head.

| Zero-based columns | Values |
|---|---|
| 0–3 | ln(RMS + 1e−9), one per channel |
| 4–7 | ln(mean absolute value + 1e−9), one per channel |
| 8–11 | ln(mean absolute adjacent-sample difference + 1e−9), one per channel |

Raw sEMG is expressed in volts. The existing extractor applies a per-record causal fourth-order Butterworth20–450Hz bandpass with `sosfilt`, then forms trailing200ms statistics. The adjacent-sample difference is not multiplied by sample rate. The exact sample-boundary convention and filter initialization are defined in the historical `semg_experiment/extract.py:make_features`; use that implementation when reproducing features. A different amplitude unit changes these logarithms and invalidates direct use of the stored normalization.

The21 feature frames span2seconds, but the causal IIR filter's support extends back to the recording start. This is not a finite2second raw-signal filter. Training queries start at t≥3seconds. Labels in the source feature file were interpolated onto the10Hz grid from angle sample times reconstructed using saved sample rates. This does not establish measured hardware clock offsets or validate synchronization for a new acquisition system.

## Checkpoint handling and reproducible example

`best.pt` contains model weights, feature mean/std, angle mean/std, selected or fixed epoch, source identities and code/protocol fingerprints. Pass unstandardized log features to the loader; do not normalize twice. This example targets the new nine-model compact bundle:

```python
import numpy as np
import torch
from load_distilled import ElbowAngleModel

torch.set_num_threads(4)
with np.load('portable_examples.npz', allow_pickle=False) as examples:
    model = ElbowAngleModel('gru/kd/best.pt')
    prediction_deg = model.predict_degrees(examples['X'])
    np.testing.assert_allclose(
        prediction_deg, examples['gru__kd'], atol=5e-4, rtol=0
    )
```

The new compact archive places these paths at its root. In the full project, final checkpoints are under `runs/student_personcv_v1/final30/FAMILY/METHOD/`, and `portable_examples.npz` is under `runs/student_personcv_v1/`. These64examples are source-training sequences used by the final refits, supplied for serialization checks; they are not an independent evaluation cohort. The legacy six-teacher archive has separate64source-record validation examples called `portable_feature_examples.npz` under `runs/teachers_source30_v1/`; those are also not independent test people. CPU thread count affects timing. The reported4thread float32 batch1 forward measurements exclude feature extraction, sensor acquisition and the initial history wait.

These30-person pretrained teachers must not initialize the later24-person fold models: they have already seen that fold's validation participants. Future teacher selection and student validation require fold-specific teachers trained on that fold's24 people.
