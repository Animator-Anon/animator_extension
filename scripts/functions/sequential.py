import math
import os

import cv2
from PIL import Image

from scripts.functions import keyframe_functions, postprocessing
from modules import processing, shared, sd_models
from modules.processing import Processed
from modules.shared import state


def main_process(myset: dict,
                 ptxt: processing.StableDiffusionProcessingTxt2Img) -> any:

    frame_count = math.ceil(myset['fps'] * myset['total_time'])
    shared.state.job_count = frame_count

    df = keyframe_functions.process_keyframes(myset)

    all_images = []

    state.job_count = frame_count

    # Need to check input source and load video if required.
    if myset['source'] == 'video':
        source_cap = cv2.VideoCapture(myset['source_file'])
    else:
        source_cap = None

    # Post Processing object dicts
    text_blocks = {}
    props = {}
    stamps = {}

    last_frame = None
    frame_save = 0

    # Main loop through frames
    for frame_no in range(frame_count):

        if state.interrupted:
            # Interrupt button pressed in WebUI
            break

        #
        # Process Keyframes
        #
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

        #
        # Get source frame
        #
        # print("Animator: Get/Generate Source Image.")


        #
        # Pre-process source frame
        #
        # print("Animator: Pre-process Source Frame.")

        #
        # Process source frame into destination frame
        #
        # print("Animator: Process Source Frame.")

        # Set prompts
        ptxt.prompt = str(df.loc[frame_no, ['pos_prompt']][0])
        ptxt.negative_prompt = str(df.loc[frame_no, ['neg_prompt']][0])

        ptxt.seed = int(df.loc[frame_no, ['seed_start']][0])
        ptxt.subseed = None \
            if df.loc[frame_no, ['seed_end']][0] is None else int(df.loc[frame_no, ['seed_end']][0])
        ptxt.subseed_strength = None \
            if df.loc[frame_no, ['seed_str']][0] is None else float(df.loc[frame_no, ['seed_str']][0])
        # print(f"Frame:{frame_no} Seed:{ptxt.seed} Sub:{ptxt.subseed} Str:{ptxt.subseed_strength}")

        ptxt.denoising_strength = df.loc[frame_no, ['denoise']][0]

        # Check if a source is set, and grab frame from there. If not, process.
        # TODO: Maybe figure out blending options for source frame and generated frame.
        if myset['source'] == 'video':
            source_cap.set(1, frame_no)
            ret, tmp_array = source_cap.read()
            post_processed_image = Image.fromarray(cv2.cvtColor(tmp_array, cv2.COLOR_BGR2RGB).astype('uint8'), 'RGB')
        elif myset['source'] == 'images':
            if frame_no >= len(source_cap):
                post_processed_image = Image.open(source_cap[-1])
                print('Out of frames, reverting to last frame!')
            else:
                post_processed_image = Image.open(source_cap[frame_no])

        else:
            processed = processing.process_images(ptxt)
            post_processed_image = processed.images[0].copy()

        if post_processed_image.mode != 'RGBA':
            post_processed_image = post_processed_image.convert('RGBA')
        #
        # Post-process destination frame
        #
        # print("Animator: Post-Process Source Frame.")
        if len(stamps) > 0:
            post_processed_image = postprocessing.paste_prop(post_processed_image,
                                                             stamps,
                                                             shared.opts.animatoranon_prop_folder)
        if len(text_blocks) > 0:
            post_processed_image = postprocessing.render_text_block(post_processed_image, text_blocks)

        #
        # Save frame
        #
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

        post_processed_image.save(os.path.join(myset['output_path'], f"frame_{frame_save:05}.png"))
        frame_save += 1

        last_frame = post_processed_image.copy()

        shared.state.current_image = post_processed_image

    Processed(ptxt, all_images, 0, "")
    print("Done.")

    return all_images
