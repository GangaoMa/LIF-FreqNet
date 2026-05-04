import numpy as np
from PIL import Image, ImageDraw


def plot_labels(img, labels, class_dict, color=(255, 0, 0), font=None):

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    for label in labels:
        draw.rectangle(label[1:], outline=color, width=2)
        draw.text((label[1], label[2] - 10), class_dict[label[0]], fill=(255, 105, 180), font=font)
    return img
