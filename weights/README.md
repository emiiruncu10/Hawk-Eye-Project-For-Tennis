# Pretrained weights

| file | source | architecture | notes |
|---|---|---|---|
| `model_best.pt` (42.9 MB) | yastrebksv/TrackNet README -> Google Drive id `1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl` | `--arch v1` (`tracknet_v1()`, 126 tensors, 10.7 M params) | Tennis-trained. Input 360x640, **BGR**, frames ordered **t, t-1, t-2**; 256-way softmax per pixel. Loads with `strict=True`, strategy `exact`. |
| `tracknetv2_track.pt` (45.4 MB) | ChgygLin/TrackNetV2-pytorch `tf2torch/track.pt` (ONNX->torch conversion of TrackNetV2 `3_in_3_out/model906_30`) | `--arch v2` layout (`conv2d_N.conv/bn` keys are renamed automatically) | **Not usable as-is**: the original Keras model normalised over the *width* axis, so every BatchNorm tensor has 512/256/128/64 entries instead of the channel count (see the repo's `tf2torch/diff.txt`). It is also a badminton (shuttlecock) model. Kept for reference only. |

Checksums: see `SHA256SUMS` (`shasum -a 256 -c SHA256SUMS`).
Weights are not committed to git (`*.pt` is ignored); re-download with the links above.
