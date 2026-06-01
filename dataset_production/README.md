# Deepfake Detection Dataset

Generated: 2026-06-01

This folder contains a validated 580-video dataset for a deepfake detection project.

## Folder Structure

```text
dataset_production/
  real/
  face_swap/
  expression_swap/
  fullbody_gan/
  metadata.csv
  rejected_videos.csv
  README.md
```

## Dataset Counts

| Category | Count |
| --- | ---: |
| real | 250 |
| face_swap | 220 |
| expression_swap | 60 |
| fullbody_gan | 50 |
| total | 580 |

## Source Summary

| Source | Categories Used | Videos | License / Terms |
| --- | --- | ---: | --- |
| godmodes/rtfs-10k | real, face_swap | 470 | CC-BY-SA-4.0 |
| hi-paris/FakeParts | fullbody_gan | 50 | CC0-1.0 |
| bitmind/FaceForensicsC23 | expression_swap | 60 | FaceForensics++ terms/citation |

Source links:

- https://huggingface.co/datasets/godmodes/rtfs-10k
- https://huggingface.co/datasets/hi-paris/FakeParts
- https://huggingface.co/datasets/bitmind/FaceForensicsC23

## Quality Rules

Accepted videos passed these checks:

- Width >= 720
- Height >= 480
- FPS >= 24
- Duration >= 1 second
- File size >= 200 KB
- Format: MP4-compatible video

Rejected videos are listed in `rejected_videos.csv`.

## Quality Summary

| Metric | Value |
| --- | ---: |
| Accepted videos | 580 |
| Rejected videos | 156 |
| Acceptance rate | 78.8% |
| Total storage used | 1.65 GB |

Resolution distribution:

| Resolution | Count |
| --- | ---: |
| 1920x1080 | 313 |
| 1280x720 | 228 |
| 720x1280 | 17 |
| 832x480 | 9 |
| 2560x1440 | 6 |
| 1906x1080 | 4 |
| 960x720 | 2 |
| 854x480 | 1 |

Codec distribution:

| Codec | Count |
| --- | ---: |
| H.264 | 360 |
| HEVC | 220 |

## Metadata

`metadata.csv` contains one row per accepted video.

Fields:

- `filename`: copied filename inside this dataset.
- `label`: one of `real`, `face_swap`, `expression_swap`, or `fullbody_gan`.
- `source`: source dataset/repository.
- `original_path`: original repository path, archive path, or shard path.
- `width`, `height`: decoded video resolution.
- `fps`: frames per second reported by OpenCV.
- `duration_s`: decoded duration in seconds.
- `file_size_kb`: copied file size in KB.
- `codec`: FourCC codec string reported by OpenCV.
- `bitrate_kbps`: estimated bitrate from file size and duration.
- `date_added`: UTC timestamp when the sample was added.
- `notes`: source-specific traceability notes.

## Rejections

`rejected_videos.csv` contains videos that were checked but not included.

Main rejection reasons:

| Reason | Count |
| --- | ---: |
| FPS too low (23.976) | 88 |
| resolution too low (640x480) | 47 |
| resolution too low (576x480) | 10 |
| resolution too low (704x480) | 4 |
| FPS too low (23.0) | 2 |
| resolution too low (656x1280) | 2 |
| FPS too low (22.182) | 1 |
| FPS too low (16.0) | 1 |
| resolution too low (608x480) | 1 |

## Citation Notes

FaceForensics++ citation:

Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images", ICCV 2019.

## Limitations

- Face-swap samples are concentrated in RTFS-10K.
- Expression-swap samples are concentrated in FaceForensics++ NeuralTextures.
- Add DFDC or Google DFD later if more source diversity is required.
