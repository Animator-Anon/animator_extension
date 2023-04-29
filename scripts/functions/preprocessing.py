import random
from PIL import Image, ImageFilter, ImageDraw, ImageOps
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


def perspective_transform(image: Image, src_points, dst_points, unsharpen):
    """
    Apply perspective transform on pillow image using transformation matrix.
    Args:
        image (PIL Image): Input image.
        src_points (list): List of four delta points from source frame.
        dst_points (list): List of four delta points from destination frame.
    Returns:
        PIL Image: Perspective transformed image.
    """
    # Convert the input image to OpenCV format.
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    ims = image.shape  # Not an image anymore, but an array

    b = np.float32([[0, 0],
                    [ims[0], 0],
                    [ims[0], ims[1]],
                    [0, ims[1]]
                   ])

    # Calculate the perspective transformation matrix.
    src_pts = np.float32(src_points) + b
    dst_pts = np.float32(dst_points) + b
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the perspective transformation.
    transformed = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

    # Convert the transformed image back to PIL format.
    timg = Image.fromarray(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))

    return timg.filter(ImageFilter.UnsharpMask(radius=2, percent=int(unsharpen)))


def old_setup_color_correction(image):
    # logging.info("Calibrating color correction.")
    correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
    return correction_target


def old_apply_color_correction(correction, original_image: Image, mask: Image):
    # logging.info("Applying color correction.")

    if mask:
        # Invert mask, use it to paste over the top after the coorection..
        backup_image = original_image.copy()

        image = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
            cv2.cvtColor(
                np.asarray(original_image),
                cv2.COLOR_RGB2LAB
            ),
            correction,
            channel_axis=2
        ), cv2.COLOR_LAB2RGB).astype("uint8"))

        # Convert grayscale image back to RGB
        #new_mask = mask.convert('RGB')

        # Combine the modified image with the backup using the alpha mask
        image = Image.composite(backup_image, image, ImageOps.invert(mask))

    else:
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