"""The pinned semantic segmenter and its policy: labels ride in the point alpha byte, unknown is never 'safe'.

SegFormer-B2 (Cityscapes, 1024 px), pinned by revision.
"""
import numpy as np

WEIGHTS_REVISION = "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"          # facebook/map-anything-apache
SEGMENTER = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
SEGMENTER_REVISION = "d633b2072669ca68d8f8e309de9b52bfdbf6bf72"
UNKNOWN_LABEL = 255
LABELS = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
          'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
MIN_LABEL_CONFIDENCE = .60


def load_segmenter(device="cuda"):
    """Semantic labels on the exact model grid. A missing segmenter means labels are unknown, never 'safe'."""
    try:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        processor = SegformerImageProcessor.from_pretrained(SEGMENTER, revision=SEGMENTER_REVISION)
        model = SegformerForSemanticSegmentation.from_pretrained(SEGMENTER, revision=SEGMENTER_REVISION).eval().to(device)
        labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
        if labels != LABELS: raise ValueError('Segmenter vocabulary/order differs from the pinned Cityscapes policy')
        return {"processor": processor, "model": model, "device": device}
    except Exception as error:
        print("Segmenter unavailable, labels will be unknown:", repr(error), flush=True)
        return None


def segment(colors, segmenter):
    """colors: HxWx3 uint8 image -> (labels uint8 HxW, score float16 HxW). Upscales to 1024 on the long side."""
    import torch
    height, width = colors.shape[:2]
    scale = 1024 / max(width, height)
    size = {"height": max(32, round(height * scale / 32) * 32), "width": max(32, round(width * scale / 32) * 32)}
    tensors = segmenter["processor"](images=colors, size=size, return_tensors="pt").to(segmenter["device"])
    with torch.inference_mode(), torch.autocast(segmenter["device"], dtype=torch.float16, enabled=segmenter["device"] == "cuda"):
        logits = segmenter["model"](**tensors).logits
    logits = torch.nn.functional.interpolate(logits.float(), size=(height, width), mode="bilinear", align_corners=False)
    score, labels = logits.softmax(dim=1).max(dim=1)
    return labels[0].cpu().numpy().astype(np.uint8), score[0].cpu().numpy().astype(np.float16)


def segment_frame(colors, valid, segmenter):
    """(labels, raw_labels, score). A failed semantic frame keeps its geometry but becomes explicitly unknown."""
    try:
        if segmenter is None: raise RuntimeError('Segmenter unavailable')
        raw_labels, score = segment(colors, segmenter)
        labels = raw_labels.copy(); labels[score < MIN_LABEL_CONFIDENCE] = UNKNOWN_LABEL
        return labels, raw_labels, score
    except Exception as failure:
        print(f'Segmentation failed for this frame, keeping its geometry: {type(failure).__name__}: {failure}', flush=True)
        unknown = np.full(valid.shape, UNKNOWN_LABEL, np.uint8)
        return unknown, unknown.copy(), np.zeros(valid.shape, np.float16)
