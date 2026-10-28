"""Image I/O helpers."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


def list_images(root):
    """All images below ``root`` (or ``[root]`` for a single file), sorted."""
    if os.path.isfile(root):
        return [root]
    paths = []
    for dirpath, _, files in sorted(os.walk(root)):
        paths += [os.path.join(dirpath, f) for f in sorted(files) if f.lower().endswith(IMG_EXTENSIONS)]
    return paths


def load_image(path_or_image, size=512, center_crop=True, multiple=64):
    """Resize the shorter side to ``size`` and center-crop to ``size x size`` (paper setting).

    With ``center_crop=False`` the aspect ratio is kept and the image is only center-cropped to a
    multiple of ``multiple`` pixels (as in the original code). ``size=0`` keeps the original scale.
    """
    image = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
    image = image.convert('RGB')
    if size:
        scale = size / min(image.size)
        image = image.resize((max(round(image.width * scale), size), max(round(image.height * scale), size)),
                             Image.BICUBIC)
    if center_crop and size:
        w = h = size
    else:
        w, h = image.width // multiple * multiple, image.height // multiple * multiple
    left, top = (image.width - w) // 2, (image.height - h) // 2
    return image.crop((left, top, left + w, top + h))


def canny_edges(image, low=50, high=100):
    """Canny edge map (3 channels, uint8) used as the ControlNet condition."""
    import cv2

    edges = cv2.Canny(np.asarray(image), low, high)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def make_comparison(images, labels=None, height=512, pad=8):
    """Concatenate images horizontally with optional captions on top."""
    images = [im.resize((round(im.width * height / im.height), height), Image.BICUBIC) for im in images]
    caption = 32 if labels else 0
    canvas = Image.new('RGB', (sum(im.width for im in images) + pad * (len(images) - 1), height + caption), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 22)
    except OSError:
        font = ImageFont.load_default()
    x = 0
    for i, im in enumerate(images):
        canvas.paste(im, (x, caption))
        if labels:
            draw.text((x + 6, 4), labels[i], fill='black', font=font)
        x += im.width + pad
    return canvas
