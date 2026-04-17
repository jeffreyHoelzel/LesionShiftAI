from typing import Any
from PIL import Image
from torchvision import transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _TransformAdapter:
    def __init__(self, tfm: T.Compose) -> None:
        self.tfm = tfm

    def __call__(self, *, image: Any) -> dict[str, Any]:
        pil_image = Image.fromarray(image)
        return {"image": self.tfm(pil_image)}


def build_train_transform(image_size: int) -> _TransformAdapter:
    tfm = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return _TransformAdapter(tfm)


def build_eval_transform(image_size: int) -> _TransformAdapter:
    tfm = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return _TransformAdapter(tfm)
