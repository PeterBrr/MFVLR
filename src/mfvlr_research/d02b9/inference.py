"""Caller-owned visual-only interface. Tested only with tiny mock modules."""
import torch

def predict_visual(model, image):
    if image.dtype != torch.float32 or image.device.type != 'cpu' or image.ndim != 4 or tuple(image.shape[1:]) != (3,224,224):
        raise ValueError('CPU float32 [B,3,224,224] required')
    if image.shape[0] not in (1,2) or not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('Invalid supplied image')
    flags = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            output = model.visual_only(image) if hasattr(model, 'visual_only') else model(image)
            logits = output.detection_logits
            masks = output.decoded.mask_logits
            if tuple(logits.shape) != (image.shape[0],2) or tuple(masks.shape) != (image.shape[0],2,224,224):
                raise ValueError('Visual output shapes')
            if not torch.isfinite(logits).all() or not torch.isfinite(masks).all():
                raise FloatingPointError('Nonfinite prediction')
            return dict(class_logits=logits, class_probabilities=logits.softmax(1),
                        mask_logits=masks, mask_probabilities=masks.softmax(1), class_order=('Real','Fake'))
    finally:
        for module, flag in flags: module.training = flag
