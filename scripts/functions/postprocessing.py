import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def paste_prop(img: Image, props: dict, prop_folder: str) -> Image:
    if img.mode != 'RGB':
        img2 = img.convert('RGBA')
    else:
        img2 = img

    for prop_name in props:
        # prop_name | prop filename | x pos | y pos | scale | rotation
        prop_filename = os.path.join(prop_folder.strip(), str(props[prop_name][1]).strip())
        x = int(props[prop_name][2])
        y = int(props[prop_name][3])
        scale = float(props[prop_name][4])
        rotation = float(props[prop_name][5])

        if not os.path.exists(prop_filename):
            print("Prop: Cannot locate file: " + prop_filename)
            return img

        prop = Image.open(prop_filename)
        w2, h2 = prop.size
        prop2 = prop.resize((int(w2 * scale), int(h2 * scale)), Image.Resampling.LANCZOS).rotate(rotation, expand=True)
        w3, h3 = prop2.size

        tmplayer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        tmplayer.paste(prop2, (int(x - w3 / 2), int(y - h3 / 2)))
        img2 = Image.alpha_composite(img2, tmplayer)

    return img2# .convert("RGB")


def render_text_block(img: Image, text_blocks: dict) -> Image:
    pad = 1  # Rounding and edge padding of the bubble background.
    d1 = ImageDraw.Draw(img)
    font_size = 20
    for text_name in text_blocks:
        # textblock_name | text_prompt | x | y | w | h | back_color | white_color | font_filename
        text_prompt = str(text_blocks[text_name][1]).strip().replace('\\n', '\n')
        x = int(text_blocks[text_name][2])
        y = int(text_blocks[text_name][3])
        w = int(text_blocks[text_name][4])
        h = int(text_blocks[text_name][5])
        # Try to convert text to a tuple (255,255,255) or just leave as text "white"
        try:
            background_colour = eval(text_blocks[text_name][6].strip())
        except:
            background_colour = text_blocks[text_name][6].strip()
        try:
            foreground_colour = eval(text_blocks[text_name][7].strip())
        except:
            foreground_colour = text_blocks[text_name][7].strip()
        font_name = str(text_blocks[text_name][8]).strip().lower()
        # Auto size the text.
        for fs in range(70):
            text_block_font = ImageFont.truetype(font_name, fs)
            txt_block_size = d1.multiline_textbbox((0, 0), text_prompt, font=text_block_font, align='center')
            if txt_block_size[2] - txt_block_size[0] > (w - pad * 2) or \
                    txt_block_size[3] - txt_block_size[1] > (h - pad * 2):
                font_size = fs - 1
                break

        text_block_font = ImageFont.truetype(font_name, font_size)
        # print(f"size:{font_size} loc:{x}, {y} size:{w}, {h}")

        txt_block_size = d1.multiline_textbbox((0, 0), text_prompt, font=text_block_font, align='center')
        # print(f"txt_block_size:{txt_block_size}")

        d1.rounded_rectangle((x, y, x + w, y + h), radius=pad, fill=background_colour)
        d1.multiline_text((x + pad, y + pad + (h - txt_block_size[3]) / 2),
                          text_prompt,
                          fill=foreground_colour,
                          font=text_block_font,
                          align='center')

    return img


def morph(img1: Image, img2: Image, count: int) -> list:
    """
    count=4
    img1:0
            0.2 (1/5)
            0.4 (2/5)
            0.6 (3/5)
            0.8 (4/5)
    img2:1
    """
    # print(f"img1: {img1.mode} img2: {img2.mode}")
    arr1 = np.array(img1).astype('float')
    diff = (np.array(img2).astype('float') - arr1) / (count+1)
    img_list = []
    for x in range(0, count):
        print(f"x: {x}")
        arr1 += diff
        img_list.append(Image.fromarray(arr1.astype('uint8'), 'RGBA'))

    return img_list
