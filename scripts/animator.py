#
# Animation Script v5.0
# Inspired by Deforum Notebook
# Must have ffmpeg installed in path.
# Poor img2img implentation, will trash images that aren't moving.
#
# See https://github.com/Animator-Anon/Animator
import json
import os
import time
import gradio as gr
import torch
import numpy as np
from scripts.functions import prepwork, sequential, loopback, export
from modules import script_callbacks, shared, sd_models, scripts, ui_common
from modules.call_queue import wrap_gradio_gpu_call
from modules.shared import cmd_opts
from modules.ui import setup_progressbar
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops


def myprocess(*args, **kwargs):
    """
    _steps
    _img_sampler_index
    _txt_sampler_index
    _width
    _height
    _cfg_scale
    _denoising_strength
    _total_time
    _fps
    _smoothing
    _film_interpolation
    _add_noise
    _noise_strength
    _seed
    _seed_travel
    _restore_faces
    _initial_img
    _loopback_mode
    _prompt_interpolation
    _tmpl_pos
    _tmpl_neg
    _key_frames
    _vid_gif
    _vid_mp4
    _vid_webm
    _style_pos
    _style_neg
    """

    i = 0
    # if the first argument is a string that says "task(...)", it is treated as a job id
    if len(args) > 0 and type(args[i]) == str and args[i][0:5] == "task(" and args[i][-1] == ")":
        i += 1

    myset = {}
    # Build a dict of the settings, so we can easily pass to sub functions.
    myset['steps'] = args[i]; i+=1                  # int(_steps),
    myset['txt_sampler_name'] = args[i]; i+=1      # int(_sampler_index)
    #myset['img_sampler_index'] = args[i]            # Need to find corresponding img2img sampler...

    from modules.sd_samplers import samplers, samplers_for_img2img
    txtsamplername = myset['txt_sampler_name']
    imgsampler = 0
    for x in range(len(samplers_for_img2img)):
        if samplers[x].name == txtsamplername:
            imgsampler = x
            break
    myset['img_sampler_name'] = samplers_for_img2img[imgsampler].name

    if (myset['img_sampler_name'] != myset['txt_sampler_name']):
        print(f"Warning: Different samplers ({myset['txt_sampler_name']}, {myset['img_sampler_name']}) have been "
              f"selected for txt2img and img2img. You probably selected one that was not allowed for img2img, so a "
              f"default was substituted.")

    myset['width'] = args[i]; i+=1                  # int(_width),
    myset['height'] = args[i]; i+=1                 # int(_height),
    myset['cfg_scale'] = args[i]; i+=1              # int(_cfg_scale),
    myset['denoising_strength'] = args[i]; i+=1     # float(_denoising_strength),
    myset['total_time'] = args[i]; i+=1             # float(_total_time),
    myset['fps'] = args[i]; i+=1                    # float(_fps),
    myset['smoothing'] = args[i]; i+=1              # int(_smoothing),
    myset['film_interpolation'] = args[i]; i+=1     # int(_film_interpolation),
    myset['add_noise'] = args[i]; i+=1              # _add_noise
    myset['noise_strength'] = args[i]; i+=1         # int(_noise_strength),
    myset['seed'] = args[i]; i+=1                   # int(_seed),
    myset['seed_travel'] = args[i]; i+=1            # bool(_seed_travel),
    myset['restore_faces'] = args[i]; i+=1          # bool(_restore_faces),
    myset['initial_img'] = args[i]; i+=1            # initial images
    myset['loopback'] = args[i]; i+=1               # bool(_loopback_mode),
    myset['prompt_interpolation'] = args[i]; i+=1   # bool(_prompt_interpolation),
    myset['tmpl_pos'] = args[i]; i+=1               # str(_tmpl_pos).strip(),
    myset['tmpl_neg'] = args[i]; i+=1               # str(_tmpl_neg).strip(),
    myset['key_frames'] = args[i]; i+=1             # str(_key_frames).strip(),
    myset['vid_gif'] = args[i]; i+=1                # bool(_vid_gif),
    myset['vid_mp4'] = args[i]; i+=1                # bool(_vid_mp4),
    myset['vid_webm'] = args[i]; i+=1               # bool(_vid_webm),
    myset['_style_pos'] = args[i]; i+=1             # str(_style_pos).strip(),
    myset['_style_neg'] = args[i]; i+=1             # str(_style_neg).strip(),
    myset['source'] = ""
    myset['debug'] = os.path.exists('debug.txt')

    print("Script Path (myprocess): ", scripts.basedir())

    # Sort out output folder
    if len(shared.opts.animatoranon_output_folder.strip()) > 0:
        output_parent_folder = shared.opts.animatoranon_output_folder.strip()
    elif myset['loopback']:
        output_parent_folder = shared.opts.outdir_img2img_samples
    else:
        output_parent_folder = shared.opts.outdir_samples
    output_parent_folder = os.path.join(output_parent_folder, time.strftime('%Y%m%d%H%M%S'))
    if not os.path.exists(output_parent_folder):
        os.makedirs(output_parent_folder)
    myset['output_path'] = output_parent_folder

    if myset['initial_img']:
        image, mask = myset['initial_img']["image"], myset['initial_img']["mask"]
        #print(f"\nif myset['initial_img']: image:{image.mode}, {image.size}")
        #print(f"\nif myset['initial_img']: mask:{mask.mode}, {mask.size}")

        # Check is mask has any data. If it does, inpant. If not, treat it like normal initial img.
        pixels = mask.convert('L').getdata()
        nMask = 0
        for pixel in pixels:
            nMask += pixel
        n = len(pixels)

        if nMask == 0:
            myset['mask'] = None
        else:
            alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
            myset['mask'] = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
            #print(f"\nif myset['initial_img']: mask:{myset['mask'].mode}, {myset['mask'].size}")

        myset['initial_img'] = image.convert("RGB")
        #print(f"\nif myset['initial_img']: image:{myset['initial_img'].mode}, {myset['initial_img'].size}")
    else:
        myset['initial_img'] = None
        myset['mask'] = None

    # Prepare the processing objects with default values.
    ptxt, pimg = prepwork.setup_processors(myset)

    # Make bat files before video incase we interrupt it, and so we can generate vids on the fly.
    export.make_batch_files(myset)

    # tmp_live_previews_enable = shared.opts.live_previews_enable
    # shared.opts.live_previews_enable = False

    shared.state.interrupted = False
    if myset['loopback']:
        result = loopback.main_process(myset, ptxt, pimg)
    else:
        result = sequential.main_process(myset, ptxt)

    if not shared.state.interrupted:
        # Generation not cancelled, go ahead and render the videos without stalling.
        export.make_videos(myset)

    shared.state.end()

    # Save the parameters to a file.
    settings_filename = os.path.join(myset['output_path'], "settings.txt")
    myset['initial_img'] = None  # No need to save the initial image
    myset['mask'] = None  # No need to save the initial image

    with open(settings_filename, "w+", encoding="utf-8") as f:
        json.dump(myset, f, ensure_ascii=False, indent=4)

    # shared.opts.live_previews_enable = tmp_live_previews_enable

    dict_str = '<ul>'
    for i in myset.keys():
        dict_str += f"<li>{i}:\t{myset[i]}</li>"
    dict_str = '</ul>'

    return result, dict_str


def ui_block_generation():
    with gr.Blocks():
        with gr.Accordion("Generation Parameters", open=False):
            gr.HTML("<p>These parameters mirror those in txt2img and img2img mode. They are used "
                    "to create the initial image in loopback mode.<br>"
                    "<b>Samplers</b>: Pick a sampler that will be used for txt2img and img2img processing. Note that "
                    "not all samplers as available for img2img, the default one will be substituted if needed.<br>"
                    "<b>Seed Travel</b>: Allow use of sub seeds to 'smoothly' change from one seed to the next. Only "
                    "makes sense to use if you manually have some seeds set in the keyframes.<br>"
                    "<b>Restore Faces</b>: Same as the checkbox in the main tabs, tries to create realistic faces... "
                    "</p>")
        steps = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=20)
        from modules.sd_samplers import samplers, samplers_for_img2img
        sampler_name = gr.Radio(label='Sampling method',
                                 choices=[x.name for x in samplers],
                                 value=samplers[0].name, type="value")

        with gr.Group():
            with gr.Row():
                width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)
                height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)
        with gr.Group():
            with gr.Row():
                cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0)
                denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01,
                                               label='Denoising strength', value=0.40)
        with gr.Row():
            seed = gr.Number(label='Seed', value=-1)
            seed_travel = gr.Checkbox(label='Seed Travel', value=False)

        with gr.Row():
            restore_faces = gr.Checkbox(label='Restore Faces', value=False)

        with gr.Row():
            with gr.Accordion("Initial Image", open=False):
                initial_img = gr.Image(label="Image for inpainting with mask",
                                       show_label=False,
                                       elem_id="aa_img2maskimg",
                                       source="upload",
                                       interactive=True,
                                       type="pil",
                                       tool="sketch",
                                       image_mode="RGBA")  # .style(height=512)

    return steps, sampler_name, width, height, cfg_scale, denoising_strength, seed, seed_travel, initial_img, \
           restore_faces


def ui_block_animation():
    with gr.Blocks():
        with gr.Accordion("Animation Parameters", open=False):
            gr.HTML("Parameters for the animation.<br>"
                    "<ul>"
                    "<li><b>Total Animation Length</b>: How long the resulting animation will be. Total number "
                    "of frames rendered will be this time * FPS</li> "
                    "<li><b>Framerate</b>: Used to calculate the number of frames, and set the rate in the output "
                    "video.</li> "
                    "<li><b>Smoothing Frames</b>: The number of additional intermediate frames to insert between "
                    "every rendered frame. These will be a faded merge of the surrounding frames.</li> "
                    "<li><b>FILM Interpolation</b>: Allow use of "
                    "<a href=\"https://github.com/google-research/frame-interpolation\"><u>FILM</u></a> to do the "
                    "interpolation, it needs to be installed separately and a bat file created so this script can call "
                    "it. Smoothing frame count is handled different by FILM. Check readme file.</li>"
                    "<li><b>Add Noise</b>: Add simple noise to the image in the form of random coloured circles. "
                    "These can help the loopback mode create new content.</li> "
                    "<li><b>Loopback Mode</b>: This is the img2img loopback mode where the resulting image, "
                    "before post processing, is pre-processed and fed back in..</li> "
                    "</ul>")
        with gr.Row():
            total_time = gr.Number(label="Total Animation Length (s)", lines=1, value=10.0)
            fps = gr.Number(label="Framerate", lines=1, value=15.0)
        with gr.Row():
            smoothing = gr.Slider(label="Smoothing_Frames", minimum=0, maximum=32, step=1, value=0)
            film_interpolation = gr.Checkbox(label="FILM Interpolation", value=False)
        with gr.Row():
            add_noise = gr.Checkbox(label="Add_Noise", value=False)
            noise_strength = gr.Slider(label="Noise Strength", minimum=0.0, maximum=1.0, step=0.01,
                                       value=0.10)
        with gr.Row():
            loopback_mode = gr.Checkbox(label='Loopback Mode', value=True)

    return total_time, fps, smoothing, film_interpolation, add_noise, noise_strength, loopback_mode


def ui_block_processing():
    with gr.Blocks():
        with gr.Accordion("Prompt Template, applied to each keyframe below", open=False):
            gr.HTML("<ul>"
                    "<li><b>Prompt interpolation:</b> Each frame's prompt will be a merge of the preceeding and "
                    "following pronmpts.</li>"
                    "<li><b>Positive / Negative Prompts</b>: Template prompts that will be applied to every frame.</li>"
                    "<li><b>Use pos prompts from style</b>: Select a pre-saved style that will be added to the "
                    "template at run time.</li>"
                    "</ul>")
        prompt_interpolation = gr.Checkbox(label='Prompt Interpolation', value=True)
        with gr.Row():
            tmpl_pos = gr.Textbox(label="Positive Prompts", lines=1, value="")
            try:
                style_pos = gr.Dropdown(label="Use pos prompts from style",
                                        choices=[k for k, v in shared.prompt_styles.styles.items()],
                                        value=next(iter(shared.prompt_styles.styles.keys())))
            except StopIteration as e:
                style_pos = gr.Dropdown(label="Use pos prompts from style",
                                        choices=[k for k, v in shared.prompt_styles.styles.items()],
                                        value=None)
        with gr.Row():
            tmpl_neg = gr.Textbox(label="Negative Prompts", lines=1, value="")
            try:
                style_neg = gr.Dropdown(label="Use neg prompts from style",
                                        choices=[k for k, v in shared.prompt_styles.styles.items()],
                                        value=next(iter(shared.prompt_styles.styles.keys())))
            except StopIteration as e:
                style_neg = gr.Dropdown(label="Use neg prompts from style",
                                        choices=[k for k, v in shared.prompt_styles.styles.items()],
                                        value=None)

    return prompt_interpolation, tmpl_pos, style_pos, tmpl_neg, style_neg


def ui_block_keyframes():
    with gr.Blocks():
        with gr.Accordion("Supported Keyframes:", open=False):
            gr.HTML("Copy and paste these templates, replace values as required.<br>"
                    "time_s | source | video, images, img2img | path<br>"
                    "time_s | prompt | positive_prompts | negative_prompts<br>"
                    "time_s | template | positive_prompts | negative_prompts<br>"
                    "time_s | prompt_from_png | file_path<br>"
                    "time_s | prompt_vtt | vtt_filepath<br>"
                    "time_s | seed | new_seed_int<br>"
                    "time_s | denoise | denoise_value<br>"
                    "time_s | cfg_scale | cfg_scale_value<br>"
                    "time_s | transform | zoom | x_shift | y_shift | rotation<br>"
                    "time_s | perspective | x0 | y0 | x1 | y1 | x2 | y2 | x3 | y3 | unsharpen_percentage<br>"
                    "time_s | noise | added_noise_strength<br>"
                    "time_s | set_text | textblock_name | text_prompt | x_pos | y_pos | width | height | fore_color |"
                    " back_color | font_name<br> "
                    "time_s | clear_text | textblock_name<br>"
                    "time_s | prop | prop_filename | x_pos | y_pos | scale | rotation<br>"
                    "time_s | set_stamp | stamp_name | stamp_filename | x_pos | y_pos | scale | rotation<br>"
                    "time_s | clear_stamp | stamp_name<br>"
                    "time_s | col_set<br>"
                    "time_s | col_clear<br>"
                    "time_s | model | " + ", ".join(
                sorted(
                    [x.model_name for x in sd_models.checkpoints_list.values()]
                )) + "</p>")

        return gr.Textbox(label="Keyframes:", lines=5, value="")


def ui_block_settings():
    with gr.Blocks():
        gr.HTML("Persistent settings moved into the main settings tab, in the group <b>Animator Extension</b>")


def ui_block_output():
    with gr.Blocks():
        with gr.Accordion("Output Block", open=False):
            gr.HTML("<p>Video creation options. Check the formats you want automatically created."
                    "Otherwise manually execute the batch files in the output folder.</p>")

        with gr.Row():
            vid_gif = gr.Checkbox(label="GIF", value=False)
            vid_mp4 = gr.Checkbox(label="MP4", value=False)
            vid_webm = gr.Checkbox(label="WEBM", value=True)

        with gr.Row():
            btn_proc = gr.Button(value="Process", variant='primary', elem_id="animator_extension_procbutton")
            btn_stop = gr.Button(value='Stop', elem_id="animator_extension_stopbutton")

        # gallery = gr.Gallery(label="gallery", show_label=True).style(grid=5, height="auto")

    return vid_gif, vid_mp4, vid_webm, btn_proc, btn_stop


#
# Basic layout of page
#
def on_ui_tabs():
    # print("on_ui_tabs")
    with gr.Blocks(analytics_enabled=False) as animator_tabs:
        with gr.Row():
            # left Column
            with gr.Column():
                with gr.Tab("Generation"):
                    steps, sampler_name, width, height, cfg_scale, denoising_strength, seed, seed_travel, image_list, \
                        restore_faces = ui_block_generation()

                    total_time, fps, smoothing, film_interpolation, add_noise, noise_strength, loopback_mode = \
                        ui_block_animation()

                    prompt_interpolation, tmpl_pos, style_pos, tmpl_neg, style_neg = ui_block_processing()

                    key_frames = ui_block_keyframes()

                with gr.Tab("Persistent Settings"):
                    ui_block_settings()

            # Right Column
            with gr.Column():
                vid_gif, vid_mp4, vid_webm, btn_proc, btn_stop = ui_block_output()

                with gr.Blocks(variant="panel"):
                    # aa_htmlinfo_x elem_id=f'html_info_x_animator_extension'
                    # aa_htmlinfo   elem_id=f'html_info_animator_extension'
                    # aa_htmllog    elem_id=f'html_log_animator_extension'
                    # aa_info       alem_id=f'generation_info_animator_extension'
                    # aa_gallery    elem_id=f"animator_extension_gallery"
                    # result_gallery, generation_info if tabname != "extras" else html_info_x, html_info, html_log
                    aa_gallery, aa_htmlinfo_x, aa_htmlinfo, aa_htmllog = \
                        ui_common.create_output_panel("animator_extension", shared.opts.animatoranon_output_folder)

        btn_proc.click(fn=wrap_gradio_gpu_call(myprocess, extra_outputs=[gr.update()]),
                       _js="start_animator",
                       inputs=[aa_htmlinfo, steps, sampler_name, width, height, cfg_scale, denoising_strength,
                               total_time, fps, smoothing, film_interpolation, add_noise, noise_strength, seed,
                               seed_travel, restore_faces, image_list, loopback_mode, prompt_interpolation,
                               tmpl_pos, tmpl_neg, key_frames, vid_gif, vid_mp4, vid_webm, style_pos, style_neg],
                       outputs=[aa_gallery, aa_htmlinfo])

        btn_stop.click(fn=lambda: shared.state.interrupt())  # ,
        # _js="reenable_animator")

    return (animator_tabs, "Animator", "animator_extension"),


#
# Define my options that will be stored in webui config
#
def on_ui_settings():
    # print("on_ui_settings")
    mysection = ('animatoranon', 'Animator Extension')

    shared.opts.add_option("animatoranon_film_folder",
                           shared.OptionInfo('C:/AI/frame_interpolation/film.bat',
                                             label="FILM batch or script file, including full path",
                                             section=mysection))
    shared.opts.add_option("animatoranon_prop_folder",
                           shared.OptionInfo('c:/ai/props',
                                             label="Prop folder",
                                             section=mysection))
    shared.opts.add_option("animatoranon_output_folder",
                           shared.OptionInfo('',
                                             label="New output folder",
                                             section=mysection))


script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
