import random
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from skimage import exposure
import cv2

def add_simple_noise(img: Image, percent: float) -> Image:
    # Draw coloured circles randomly over the image. Lame, but for testing.
    # print("Noise function")
    w2, h2 = img.size
    draw = ImageDraw.Draw(img)
    for i in range(int(50 * float(percent))):
        x2 = random.randint(0, w2)
        y2 = random.randint(0, h2)
        s2 = random.randint(0, int(50 * float(percent)))
        pos = (x2, y2, x2 + s2, y2 + s2)
        draw.ellipse(pos, fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                     outline=(0, 0, 0))
    return img


def transform_image(img: Image, rot: float, x: int, y: int, zoom: float) -> Image:
    w, h = img.size

    # Zoom image
    img2 = img.resize((int(w * zoom), int(h * zoom)), Image.Resampling.LANCZOS)

    # Create background image
    padding = 2
    resimg = add_simple_noise(img.copy(), 0.75).resize((w + padding * 2, h + padding * 2), Image.Resampling.LANCZOS). \
        filter(ImageFilter.GaussianBlur(5)). \
        crop((padding, padding, w + padding, h + padding))

    resimg.paste(img2.rotate(rot), (int((w - img2.size[0]) / 2 + x), int((h - img2.size[1]) / 2 + y)))

    return resimg

def old_setup_color_correction(image):
    # logging.info("Calibrating color correction.")
    correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
    return correction_target


def old_apply_color_correction(correction, original_image):
    # logging.info("Applying color correction.")
    image = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
        cv2.cvtColor(
            np.asarray(original_image),
            cv2.COLOR_RGB2LAB
        ),
        correction,
        channel_axis=2
    ), cv2.COLOR_LAB2RGB).astype("uint8"))

    # This line breaks it
    # image = blendLayers(image, original_image, BlendType.LUMINOSITY)

    return image