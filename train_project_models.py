"""Train the missing VDS classifier artifacts from the course data packs.

Outputs:
    weights/color_model.pth
    weights/type_model.pth
    weights/brand_model.pth
    weights/brand_labels.txt

The source data stays in the original course folder; this script reads the zip
files directly so the repository does not need a large extracted dataset.
"""

from __future__ import annotations

import argparse
import io
import random
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from brand_classify import BrandClassifier
from color_classify import ColorClassifier, COLOR_LABELS
from type_classify import TYPE_LABELS, TypeClassifier


DEFAULT_DATA_ROOT = Path(
    r"D:\OneDrive - Southern Cross University\桌面\大作业\基于车辆特征分析的套牌车稽查系统"
)
WEIGHTS_DIR = Path("weights")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classifier_transform(train: bool) -> transforms.Compose:
    steps = [
        transforms.Resize((224, 224)),
    ]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(steps)


class NestedZipImageDataset(Dataset):
    """Read images from an outer zip that contains an inner images.zip."""

    def __init__(
        self,
        outer_zip: Path,
        class_to_idx: dict[str, int],
        transform: transforms.Compose,
        aliases: dict[str, str] | None = None,
    ) -> None:
        self.outer_zip = outer_zip
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.aliases = aliases or {}

        with zipfile.ZipFile(outer_zip) as outer:
            inner_name = next(n for n in outer.namelist() if n.endswith("images.zip"))
            self.inner_bytes = outer.read(inner_name)

        with zipfile.ZipFile(io.BytesIO(self.inner_bytes)) as inner:
            samples: list[tuple[str, int]] = []
            for name in inner.namelist():
                if not name.lower().endswith(IMAGE_SUFFIXES):
                    continue
                parts = name.split("/")
                if len(parts) < 3:
                    continue
                label = self.aliases.get(parts[-2], parts[-2])
                if label in class_to_idx:
                    samples.append((name, class_to_idx[label]))
            self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        name, label = self.samples[index]
        with zipfile.ZipFile(io.BytesIO(self.inner_bytes)) as inner:
            image = Image.open(io.BytesIO(inner.read(name))).convert("RGB")
        return self.transform(image), label


@dataclass(frozen=True)
class BrandSample:
    image_name: str
    label: str
    bbox: tuple[int, int, int, int]


class BrandZipDataset(Dataset):
    """Read brand crops from the Pascal VOC zip file."""

    def __init__(
        self,
        zip_path: Path,
        samples: list[BrandSample],
        label_to_idx: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.zip_path = zip_path
        self.samples = samples
        self.label_to_idx = label_to_idx
        self.transform = transform

        with zipfile.ZipFile(zip_path) as zf:
            image_names = [
                n for n in zf.namelist() if "/img/" in n and n.lower().endswith(IMAGE_SUFFIXES)
            ]
        self.image_lookup = {Path(n).name: n for n in image_names}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image_path = self.image_lookup[sample.image_name]
        with zipfile.ZipFile(self.zip_path) as zf:
            image = Image.open(io.BytesIO(zf.read(image_path))).convert("RGB")
        x1, y1, x2, y2 = sample.bbox
        image = image.crop((x1, y1, x2, y2))
        return self.transform(image), self.label_to_idx[sample.label]


def parse_brand_samples(zip_path: Path) -> tuple[list[BrandSample], list[str]]:
    samples: list[BrandSample] = []
    labels: set[str] = set()
    with zipfile.ZipFile(zip_path) as zf:
        xml_names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        for xml_name in xml_names:
            root = ET.fromstring(zf.read(xml_name))
            filename = root.findtext("filename")
            if not filename:
                continue
            for obj in root.findall("object"):
                label = (obj.findtext("name") or "").strip()
                box = obj.find("bndbox")
                if not label or box is None:
                    continue
                try:
                    x1 = int(float(box.findtext("xmin", "0")))
                    y1 = int(float(box.findtext("ymin", "0")))
                    x2 = int(float(box.findtext("xmax", "0")))
                    y2 = int(float(box.findtext("ymax", "0")))
                except ValueError:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                labels.add(label)
                samples.append(BrandSample(filename, label, (x1, y1, x2, y2)))
    return samples, sorted(labels)


def train_model(
    name: str,
    model: nn.Module,
    dataset: Dataset,
    output_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_ratio: float,
    seed: int,
) -> float:
    if len(dataset) < 2:
        raise RuntimeError(f"{name}: dataset is too small ({len(dataset)} samples)")

    gen = torch.Generator().manual_seed(seed)
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    dev = device()
    model.to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_acc = -1.0

    print(f"\n[{name}] samples={len(dataset)} train={train_size} val={val_size} device={dev}")
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(dev)
            labels = labels.to(dev)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += outputs.argmax(1).eq(labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(dev)
                labels = labels.to(dev)
                outputs = model(inputs)
                val_correct += outputs.argmax(1).eq(labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        if val_acc > best_acc:
            best_acc = val_acc
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)

        avg_loss = train_loss / max(len(train_loader), 1)
        mark = "*" if val_acc >= best_acc else ""
        print(
            f"[{name}] epoch {epoch + 1:02d}/{epochs} "
            f"loss={avg_loss:.4f} train_acc={train_acc:.2%} val_acc={val_acc:.2%} {mark}"
        )

    print(f"[{name}] saved {output_path} best_val_acc={best_acc:.2%}")
    return best_acc


def train_color(data_root: Path, epochs: int, batch_size: int, seed: int) -> float:
    class_to_idx = {COLOR_LABELS[i]: i for i in sorted(COLOR_LABELS)}
    dataset = NestedZipImageDataset(
        data_root / "任务8生成的车辆颜色识别数据.zip",
        class_to_idx=class_to_idx,
        transform=classifier_transform(train=True),
    )
    return train_model(
        "color",
        ColorClassifier(num_classes=len(class_to_idx)),
        dataset,
        WEIGHTS_DIR / "color_model.pth",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=5e-4,
        val_ratio=0.2,
        seed=seed,
    )


def train_type(data_root: Path, epochs: int, batch_size: int, seed: int) -> float:
    class_to_idx = {TYPE_LABELS[i]: i for i in sorted(TYPE_LABELS)}
    dataset = NestedZipImageDataset(
        data_root / "任务4生成的车辆类型数据.zip",
        class_to_idx=class_to_idx,
        aliases={"minibus": "mini"},
        transform=classifier_transform(train=True),
    )
    return train_model(
        "type",
        TypeClassifier(num_classes=len(class_to_idx)),
        dataset,
        WEIGHTS_DIR / "type_model.pth",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=5e-4,
        val_ratio=0.2,
        seed=seed,
    )


def train_brand(data_root: Path, epochs: int, batch_size: int, seed: int) -> float:
    brand_zip = data_root / "阶段三-任务4资料.zip"
    samples, labels = parse_brand_samples(brand_zip)
    label_path = WEIGHTS_DIR / "brand_labels.txt"
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    dataset = BrandZipDataset(
        brand_zip,
        samples=samples,
        label_to_idx=label_to_idx,
        transform=classifier_transform(train=True),
    )
    return train_model(
        "brand",
        BrandClassifier(num_classes=len(labels)),
        dataset,
        WEIGHTS_DIR / "brand_model.pth",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=3e-4,
        val_ratio=0.2,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--target", choices=["all", "color", "type", "brand"], default="all")
    parser.add_argument("--color-epochs", type=int, default=8)
    parser.add_argument("--type-epochs", type=int, default=10)
    parser.add_argument("--brand-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    print(f"data_root={args.data_root}")
    print(f"device={device()}")

    if args.target in ("all", "color"):
        train_color(args.data_root, args.color_epochs, args.batch_size, args.seed)
    if args.target in ("all", "type"):
        train_type(args.data_root, args.type_epochs, args.batch_size, args.seed)
    if args.target in ("all", "brand"):
        train_brand(args.data_root, args.brand_epochs, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
