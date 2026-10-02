# Video to 3D

Turn overlapping walking videos into one colored 3D point cloud. The pipeline estimates depth with MapAnything, aligns captures using their camera poses and GPS, refines geometry against LiDAR, and fuses overlapping observations into a single scene.

**Current input:** ARKit video captures with synchronized camera metadata and classified LiDAR. Plain video files without that metadata are not supported. The output is a point cloud, not a textured mesh.

## Requirements

- Linux, Python 3.10+, Git, and FFmpeg on your `PATH`.
- An NVIDIA GPU with a compatible CUDA-enabled PyTorch installation for inference. CPU inference is available with `--device cpu`, but is substantially slower.
- Overlapping captures of the same area, plus classified LiDAR covering that area.

The current geographic frame is fixed to Ithaca: EPSG:6347, projected origin `[377249.9995, 4700249.9995, 150.0]`, with model axes east, up, south in metres. Captures and LiDAR must match this frame; another region requires adapting the coordinate constants and projection in the stages and exporter.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) if your GPU needs a specific CUDA build. MapAnything source is pinned in `requirements.txt`; MapAnything and SegFormer weights are pinned in `pipeline/stages/segment.py` and downloaded on first use. Set `HF_HOME` to choose their cache directory.

## Prepare captures

Give each capture its own folder:

```text
captures/
  capture-01/
    capture.mov
    manifest.json
  capture-02/
    capture.mov
    manifest.json
lidar/
  classified_<tile>.laz
```

`video/capture.mov` inside a capture folder is also accepted. Capture IDs come from each manifest and must be unique. The pipeline reads the original videos without modifying them.

Each `manifest.json` must contain:

| Field | Contents |
| --- | --- |
| `id`, `created_at` | Capture identifier and creation timestamp |
| `session_started_timestamp` | Session start in the same time base as frame timestamps |
| `video` | `duration_s` and, preferably, `frame_count` for checking video/metadata alignment |
| `frames` | Keyframes with the fields below |

Every keyframe needs `video_frame_index` (zero-based), `video_timestamp_s`, `timestamp`, `transform` (4×4 ARKit camera-to-world matrix), `intrinsics` (3×3), `calibration_image_dimensions` (`[width, height]`), `tracking_state`, `feature_point_count`, and `feature_points.points` (3D ARKit feature positions). GPS is supplied as `location` with `latitude`, `longitude`, and `horizontal_accuracy_m`.

Optional `pose_samples` entries contain `timestamp` and `transform` for detecting tracking drift between keyframes. Optional `exif_numeric` exposure/ISO fields help reject blurred or dark frames. Usable keyframes need normal tracking and nonempty feature points. Each capture needs at least 15 seconds, approximately 20 metres of GPS coverage, and enough usable keyframes to form chunks of at least six frames.

LiDAR files must be named `classified_*.laz`, with projected coordinates in EPSG:6347 and compatible metre-based elevations (the existing reference uses NAVD88). Placement uses ground classes 2 and 17; building refinement uses class 6. LiDAR classification is not included. Optional raw `u_*.laz` tiles add a separate ground reference layer to the exported model.

## Reconstruct

```bash
python pipeline/reconstruct.py captures --lidar lidar --output output
```

You can also pass one capture folder. Completed captures are cached under the output directory and reused on subsequent runs. Use a separate output directory for each scene.

```bash
# Reconstruct one capture again, then merge it with the other cached captures.
python pipeline/reconstruct.py captures --lidar lidar --output output \
  --only <capture-id-or-unique-prefix> --force

# Reduce frames per chunk when GPU memory is limited.
python pipeline/reconstruct.py captures --lidar lidar --frames-per-chunk 20
```

`--volume-workers` controls fusion parallelism (default 4). `--amp-dtype fp16` selects half precision for GPUs without bfloat16 support. See `--help` for all options.

The pipeline extracts frames, rejects unstable tracking, groups frames into overlapping 30-metre chunks, estimates depth and semantic labels, aligns geometry to GPS and LiDAR, then fuses the captures and exports the model. Failed captures retain their intermediates under `output/work/`. A successful rebuild preserves the previous model as `output/model.previous/`.

## Output

| Path | Contents |
| --- | --- |
| `output/model/preview.ply` | Colored point cloud for point-cloud viewers and 3D tools |
| `output/model/model.json` | Bounds, semantic classes, capture summaries, tile index, and download checksum |
| `output/model/coordinates.json` | Origin, projection, axes, and units |
| `output/model/tiles/` | Coarse preview and full-resolution binary point tiles |
| `output/model/lidar-coarse.bin` | Optional LiDAR reference, excluded from the PLY and model point count |
| `output/walks/` | Cached per-capture reconstructions |
| `output/merged/` | Fused intermediate model for re-exporting |

Binary point records are 16 bytes: three little-endian float32 coordinates, three uint8 RGB channels, and one uint8 semantic class. The PLY contains coordinates and RGB only.

To export an existing merged model again:

```bash
python pipeline/export.py --merged output/merged --output output/model-export
```

Use a new export directory: the exporter replaces its destination. Its input is the intermediate `merged/` directory, not an already exported `model/` directory.

## Code

`pipeline/reconstruct.py` runs reconstruction, `pipeline/stages/` implements geometry and inference, and `pipeline/export.py` writes the portable model files.

Project code is [MIT licensed](LICENSE). [MapAnything](https://github.com/facebookresearch/map-anything) and [SegFormer](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024) have their own model and dependency licenses.
