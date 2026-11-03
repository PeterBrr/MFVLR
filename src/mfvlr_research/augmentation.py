"""Train-only augmentation following DeepfakeBench's UCF recipe (commit f188b1c).

Parameters come from training/config/detector/ucf.yaml and the pipeline order from
init_data_aug_method(); this is an OpenCV rewrite, not the Albumentations code, so
values match but draws do not. Each draw derives from (seed, update, sample), so a
resumed run repeats the same transforms without storing any generator state.
"""
import cv2
import numpy as np
import torch

cv2.setNumThreads(1)


def _rotate(image, mask, angle):
    height, width = mask.shape
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1.0)
    return (cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101),
            cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101))


def _brightness_contrast(image, rng):
    alpha, beta = 1 + rng.uniform(-.1, .1), rng.uniform(-.1, .1) * 255
    return np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def _fancy_pca(image, rng):
    pixels = image.reshape(-1, 3).astype(np.float64) / 255
    values, vectors = np.linalg.eigh(np.cov(pixels - pixels.mean(0), rowvar=False))
    shift = vectors @ (rng.normal(0, .1, 3) * values)
    return (np.clip(image.astype(np.float64) / 255 + shift, 0, 1) * 255).astype(np.uint8)


def _hue_saturation_value(image, rng):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int32)
    hsv[..., 0] = (hsv[..., 0] + int(rng.uniform(-20, 20))) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + int(rng.uniform(-30, 30)), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + int(rng.uniform(-20, 20)), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _jpeg(image, quality):
    ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (cv2.IMWRITE_JPEG_QUALITY, quality))
    if not ok:
        raise ValueError('JPEG encoding failed')
    return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def augment_sample(sample, mode, seed, update, index):
    """Return a transformed copy; the flip decision is shared by both modes for one draw key."""
    rng = np.random.default_rng([seed, update, index])
    image = sample['image'].mul(255).round().byte().permute(1, 2, 0).numpy().copy()
    mask = sample['mask'].numpy().astype(np.uint8).copy()
    if rng.random() < .5:
        image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
    if mode == 'deepfakebench':
        if rng.random() < .5:
            image, mask = _rotate(image, mask, rng.uniform(-10, 10))
        if rng.random() < .5:
            size = int(rng.choice((3, 5, 7)))
            image = cv2.GaussianBlur(image, (size, size), 0)
        if rng.random() < .5:
            color = (_brightness_contrast, _fancy_pca, _hue_saturation_value)[int(rng.integers(3))]
            image = color(image, rng)
        if rng.random() < .5:
            image = _jpeg(image, int(rng.integers(40, 101)))
    elif mode != 'flip':
        raise ValueError(f'Unknown augmentation {mode}')
    return {**sample, 'image': torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255),
            'mask': torch.from_numpy(mask.copy()).long()}
