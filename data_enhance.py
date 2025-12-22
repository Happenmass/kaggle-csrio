import albumentations as A
import cv2
from pathlib import Path
from albumentations.pytorch import ToTensorV2

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
SIZE = 256

# 训练：tile 内激进裁切 + 颜色/成像退化（但最终严格对齐 cfg）
train_tile_tfms_stable = A.Compose([
    # 在 tile 内做更激进的裁切，然后输出到 256×256
    A.RandomResizedCrop(
        size=(SIZE, SIZE),
        scale=(0.55, 1.0),          # 可改为 (0.35, 1.0) 更激进
        ratio=(0.9, 1.1),
        interpolation=cv2.INTER_CUBIC,
        p=1.0
    ),

    # 方向增强（草地通常方向不敏感）
    A.OneOf([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.RandomRotate90(p=1.0),
    ], p=0.8),

    # 光照/颜色（幅度中等，别把绿草变异色）
    A.RandomBrightnessContrast(0.15, 0.15, p=0.6),
    A.HueSaturationValue(8, 12, 8, p=0.4),
    A.RandomGamma(gamma_limit=(85, 115), p=0.25),

    # 成像退化（轻度）
    A.OneOf([
        A.ImageCompression(quality_range=(55, 95)),
        A.GaussNoise(std_range=(0.02, 0.10)),
        A.GaussianBlur(blur_limit=(3, 5)),
        A.MotionBlur(blur_limit=3),
    ], p=0.30),

    # 与 cfg 对齐的归一化
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

train_tile_tfms_extreme = A.Compose([
    # 在 tile 内做更激进的裁切，然后输出到 256×256
    A.RandomResizedCrop(
        size=(SIZE, SIZE),
        scale=(0.35, 1.0),          # 可改为 (0.35, 1.0) 更激进
        ratio=(0.9, 1.1),
        interpolation=cv2.INTER_CUBIC,
        p=1.0
    ),

    # 方向增强（草地通常方向不敏感）
    A.OneOf([
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.RandomRotate90(p=1.0),
    ], p=0.8),

    # 光照/颜色（幅度中等，别把绿草变异色）
    A.RandomBrightnessContrast(0.15, 0.15, p=0.6),
    A.HueSaturationValue(8, 12, 8, p=0.4),
    A.RandomGamma(gamma_limit=(85, 115), p=0.25),

    # 成像退化（轻度）
    A.OneOf([
        A.ImageCompression(quality_range=(55, 95)),
        A.GaussNoise(std_range=(0.02, 0.10)),
        A.GaussianBlur(blur_limit=(3, 5)),
        A.MotionBlur(blur_limit=3),
    ], p=0.30),

    # 与 cfg 对齐的归一化
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

# 验证/推理：严格按 cfg 的“中心裁切”（crop_pct=1.0 等价于不额外裁，只做 resize）
# 因为你 tile 不是方的，这里我建议直接 resize 到 256×256（bicubic）
# 这与 cfg 的 fixed_input_size=false + input_size=256 的惯例是一致的。
val_tile_tfms = A.Compose([
    A.Resize(SIZE, SIZE, interpolation=cv2.INTER_CUBIC),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])


def _tensor_to_uint8_img(tensor):
    """Convert normalized CHW tensor to uint8 HWC RGB image."""
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected torch.Tensor from albumentations pipeline")
    img = tensor.detach().cpu().float().permute(1, 2, 0).numpy()  # HWC, RGB, normalized
    img = img * STD + MEAN  # de-normalize
    img = (img * 255.0).clip(0, 255).astype("uint8")
    return img


def _load_images(input_path):
    """Yield (Path, rgb_image_np) for each image."""
    import numpy as np

    path = Path(input_path)
    files = []
    if path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        files = [p for p in path.iterdir() if p.suffix.lower() in exts and p.is_file()]
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(f"{input_path} not found")

    for file in files:
        bgr = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        yield file, rgb


def _get_transform(name: str):
    name = name.lower()
    if name == "stable":
        return train_tile_tfms_stable
    if name == "extreme":
        return train_tile_tfms_extreme
    if name == "val":
        return val_tile_tfms
    raise ValueError(f"unknown transform '{name}' (choose stable|extreme|val)")


def _save_augmented_images(images, tfm, out_dir: Path, samples: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    for src_path, rgb in images:
        base = src_path.stem
        for i in range(samples):
            augmented = tfm(image=rgb)["image"]
            img_uint8 = _tensor_to_uint8_img(augmented)
            bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            out_file = out_dir / f"{base}_aug{i+1}.jpg"
            cv2.imwrite(str(out_file), bgr)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Preview data augmentations and save outputs.")
    p.add_argument("--input", required=True, help="image file or directory containing images")
    p.add_argument("--output-dir", default="aug_preview", help="where augmented images are written")
    p.add_argument("--transform", default="stable", help="stable | extreme | val")
    p.add_argument("--samples", type=int, default=3, help="number of augmented variants per image")
    args = p.parse_args()

    transform = _get_transform(args.transform)
    imgs = list(_load_images(args.input))
    if not imgs:
        raise SystemExit("no images found; check --input path")
    _save_augmented_images(imgs, transform, Path(args.output_dir), samples=max(1, args.samples))
    print(f"Saved {len(imgs) * max(1, args.samples)} images to {args.output_dir}")
