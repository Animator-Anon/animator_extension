import math
import os
import cv2

from scripts.functions import keyframe_functions, preprocessing, postprocessing
from modules import processing, shared, sd_models
from modules.processing import Processed
from modules.shared import state
from PIL import Image, ImageEnhance
import glob


def main_process(myset: dict,
                 ptxt: processing.StableDiffusionProcessingTxt2Img,
                 pimg: processing.StableDiffusionProcessingImg2Img) -> any:
    apply_colour_corrections = True
    x_shift_cumulative = 0
    y_shift_cumulative = 0

    frame_count = math.ceil(myset['fps'] * myset['total_time'])
    state.job_count = frame_count

    df = keyframe_functions.process_keyframes(myset)

    all_images = []

    # Post Processing object dicts
    text_blocks = {}
    props = {}
    stamps = {}

    last_frame = None
    frame_save = 0

    ptxt.seed = -1
    processing.fix_seed(ptxt)

    pimg.batch_size = 1
    pimg.n_iter = 1
    pimg.do_not_save_grid = True

    initial_color_corrections = None

    # Need to check input source and load video if required.
    if myset['source'] == 'video':
        source_cap = cv2.VideoCapture(myset['source_file'])
    elif myset['source'] == 'images':
        source_cap = myset['source_file']
    else:
        source_cap = None

    # Main loop through frames
    for frame_no in range(frame_count):

        if state.interrupted:
            # Interrupt button pressed in WebUI
            break

        #############################
        # Process Keyframes
        #############################
        # Check if keyframes exists for this frame
        # print("process keyframes")
        if frame_no in myset['keyframes']:
            # Keyframes exist for this frame.
            # print(f"\r\nKeyframe at {frame_no}: {myset['keyframes'][frame_no]}\r\n")

            for keyframe in myset['keyframes'][frame_no]:
                keyframe_command = keyframe[0].lower().strip()
                # Check the command, should be first item.
                if keyframe_command == "model" and len(keyframe) == 2:
                    # Time (s) | model    | model name
                    info = sd_models.get_closet_checkpoint_match(keyframe[1].strip() + ".ckpt")
                    if info is None:
                        raise RuntimeError(f"Unknown checkpoint: {keyframe[1]}")
                    sd_models.reload_model_weights(shared.sd_model, info)

                elif keyframe_command == "col_set" and len(keyframe) == 1:
                    # Time (s) | col_set
                    apply_colour_corrections = True
                    if frame_no > 0:
                        # Colour correction is set automatically above
                        initial_color_corrections = [processing.setup_color_correction(pimg.init_images[0])]
                elif keyframe_command == "col_clear" and len(keyframe) == 1:
                    # Time (s) | col_clear
                    apply_colour_corrections = False

                elif keyframe_command == "prop" and len(keyframe) == 6:
                    # Time (s) | prop | prop_filename | x pos | y pos | scale | rotation
                    # bit of a hack, no prop name is supplied, but same function is used to draw.
                    # so the command is passed in place of prop name, which will be ignored anyway.
                    props[len(props)] = keyframe
                elif keyframe_command == "set_stamp" and len(keyframe) == 7:
                    # Time (s) | set_stamp | stamp_name | stamp_filename | x pos | y pos | scale | rotation
                    stamps[keyframe[1].strip()] = keyframe[1:]
                elif keyframe_command == "clear_stamp" and len(keyframe) == 2:
                    # Time (s) | clear_stamp | stamp_name
                    if keyframe[1].strip() in stamps:
                        stamps.pop(keyframe[1].strip())

                elif keyframe_command == "set_text" and len(keyframe) == 10:
                    # time_s | set_text | name | text_prompt | x | y | w | h | fore_color | back_color | font_name
                    text_blocks[keyframe[1].strip()] = keyframe[1:]
                elif keyframe_command == "clear_text" and len(keyframe) == 2:
                    # Time (s) | clear_text | textblock_name
                    if keyframe[1].strip() in text_blocks:
                        text_blocks.pop(keyframe[1].strip())

        #############################
        # Get source frame
        #############################
        # print("Animator: Get/Generate Source Image.")
        if myset['source'] == 'video':
            source_cap.set(1, frame_no)
            ret, tmp_array = source_cap.read()
            init_img = Image.fromarray(cv2.cvtColor(tmp_array, cv2.COLOR_BGR2RGB).astype('uint8'), 'RGB')
        elif myset['source'] == 'images':
            if frame_no >= len(source_cap):
                init_img = Image.open(source_cap[-1])
                print('Out of frames, reverting to last frame!')
            else:
                init_img = Image.open(source_cap[frame_no])
        elif frame_no == 0:
            # Generate initial image
            print(f"Initial Image: {myset['initial_img']}")
            if myset['initial_img'] is None:
                ptxt.prompt = str(df.loc[0, ['pos_prompt']][0])
                ptxt.negative_prompt = str(df.loc[0, ['neg_prompt']][0])
                init_processed = processing.process_images(ptxt)
                init_img = init_processed.images[0]
            else:
                init_img = myset['initial_img']
                if init_img.size != (myset['width'], myset['height']):
                    init_img = init_img.resize((myset['width'], myset['height']), Image.Resampling.LANCZOS)
        else:
            init_img = last_frame

        if init_img.mode != 'RGBA':
            init_img = init_img.convert('RGBA')

        if frame_no == 0:
            initial_color_corrections = preprocessing.old_setup_color_correction(init_img)
            # [processing.setup_color_correction(init_img)]

        ############################
        # Pre-process source frame
        ############################
        # print("Animator: Pre-process Source Frame.")
        # Update transform details
        x_shift_per_frame = df.loc[frame_no, ['x_shift']][0]
        y_shift_per_frame = df.loc[frame_no, ['y_shift']][0]
        rot_per_frame = df.loc[frame_no, ['rotation']][0]
        zoom_factor = df.loc[frame_no, ['zoom']][0]

        # Translate source frame when source is img2img where they have an effect frame to frame.
        x_shift_cumulative = x_shift_cumulative + x_shift_per_frame
        y_shift_cumulative = y_shift_cumulative + y_shift_per_frame

        if x_shift_per_frame != 0 or y_shift_per_frame != 0 or rot_per_frame != 0 or zoom_factor != 1.0:
            init_img = preprocessing.transform_image(init_img, rot_per_frame, int(x_shift_cumulative),
                                                     int(y_shift_cumulative), zoom_factor)

        # Subtract the integer portion we just shifted.
        x_shift_cumulative = x_shift_cumulative - int(x_shift_cumulative)
        y_shift_cumulative = y_shift_cumulative - int(y_shift_cumulative)

        # Perspective transform
        if  df.loc[frame_no, ['px0']][0] != 0 or df.loc[frame_no, ['py0']][0] != 0 or \
            df.loc[frame_no, ['px1']][0] != 0 or df.loc[frame_no, ['py1']][0] != 0 or \
            df.loc[frame_no, ['px2']][0] != 0 or df.loc[frame_no, ['py2']][0] != 0 or \
            df.loc[frame_no, ['px3']][0] != 0 or df.loc[frame_no, ['py3']][0] != 0:

            init_img = \
                preprocessing.perspective_transform(init_img,
                                                    [(df.loc[frame_no, ['px0']][0], df.loc[frame_no, ['py0']][0]),
                                                     (df.loc[frame_no, ['px1']][0], df.loc[frame_no, ['py1']][0]),
                                                     (df.loc[frame_no, ['px2']][0], df.loc[frame_no, ['py2']][0]),
                                                     (df.loc[frame_no, ['px3']][0], df.loc[frame_no, ['py3']][0])],
                                                    [(0, 0), (0, 0), (0, 0), (0, 0)],
                                                    df.loc[frame_no, ['punsharpen']][0])

        # Props
        if len(props) > 0:
            # print("Pasting prop into image.")
            init_img = postprocessing.paste_prop(init_img, props, shared.opts.animatoranon_prop_folder)
            props = {}

        # Noise
        if myset['add_noise']:
            # print("Adding Noise!!")
            init_img = preprocessing.add_simple_noise(init_img, df.loc[frame_no, ['noise']][0])

        #############################
        # Process source frame into destination frame
        #############################
        # print("Animator: Process Source Frame.")
        if apply_colour_corrections:
            init_img = preprocessing.old_apply_color_correction(initial_color_corrections, init_img)

        # Set prompts
        pimg.prompt = str(df.loc[frame_no, ['pos_prompt']][0])
        pimg.negative_prompt = str(df.loc[frame_no, ['neg_prompt']][0])

        pimg.seed = int(df.loc[frame_no, ['seed_start']][0])
        pimg.subseed = None \
            if df.loc[frame_no, ['seed_end']][0] is None else int(df.loc[frame_no, ['seed_end']][0])
        pimg.subseed_strength = None \
            if df.loc[frame_no, ['seed_str']][0] is None else float(df.loc[frame_no, ['seed_str']][0])
        # print(f"Frame:{frame_no} Seed:{pimg.seed} Sub:{pimg.subseed} Str:{pimg.subseed_strength}")

        pimg.denoising_strength = df.loc[frame_no, ['denoise']][0]

        pimg.init_images = [init_img]

        processed = processing.process_images(pimg)

        #############################
        # Post-process destination frame
        #############################
        # print("Animator: Post-Process Source Frame.")
        post_processed_image = processed.images[0].copy()
        if post_processed_image.mode != 'RGBA':
            post_processed_image = post_processed_image.convert('RGBA')

        if len(stamps) > 0:
            post_processed_image = postprocessing.paste_prop(post_processed_image,
                                                             stamps,
                                                             shared.opts.animatoranon_prop_folder)
        if len(text_blocks) > 0:
            post_processed_image = postprocessing.render_text_block(post_processed_image, text_blocks)

        #############################
        # Save frame
        #############################
        # Create and save smoothed intermediate frames
        if frame_no > 0 and myset['smoothing'] > 0 and not myset['film_interpolation']:
            # working a frame behind, smooth from last_frame -> post_processed_image
            for idx, img in enumerate(postprocessing.morph(last_frame, post_processed_image, myset['smoothing'])):
                img.save(os.path.join(myset['output_path'], f"frame_{frame_save:05}.png"))
                print(f"{frame_save:03}: {frame_no:03} > {idx} smooth frame")
                frame_save += 1

        # print("Animator: Save Frame")
        if frame_no % int(myset['fps']) == 0:
            all_images.append(post_processed_image)

        # don't post process the loopback frame.
        last_frame = processed.images[0]
        if last_frame.mode != 'RGBA':
            last_frame = last_frame.convert('RGBA')

        post_processed_image.save(os.path.join(myset['output_path'], f"frame_{frame_save:05}.png"))
        frame_save += 1

        shared.state.current_image = post_processed_image

    Processed(pimg, all_images, 0, "")
    print("Done.")

    return all_images
