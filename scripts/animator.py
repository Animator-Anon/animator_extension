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
from scripts.functions import prepwork, sequential, loopback, export
from modules import script_callbacks, shared, sd_models, scripts, ui_common
from modules.call_queue import wrap_gradio_gpu_call
from modules.shared import cmd_opts
from modules.ui import setup_progressbar


def myprocess(*args, **kwargs):
    """
    _steps
    _sampler_index
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

    # Build a dict of the settings, so we can easily pass to sub functions.
    myset = {'steps': args[i + 0],  # int(_steps),
             'sampler_index': args[i + 1],  # int(_sampler_index),
             'width': args[i + 2],  # int(_width),
             'height': args[i + 3],  # int(_height),
             'cfg_scale': args[i + 4],  # float(_cfg_scale),
             'denoising_strength': args[i + 5],  # float(_denoising_strength),
             'total_time': args[i + 6],  # float(_total_time),
             'fps': args[i + 7],  # float(_fps),
             'smoothing': args[i + 8],  # int(_smoothing),
             'film_interpolation': args[i + 9],  # int(_film_interpolation),
             'add_noise': args[i + 10],  # _add_noise,
             'noise_strength': args[i + 11],  # float(_noise_strength),
             'seed': args[i + 12],  # int(_seed),
             'seed_travel': args[i + 13],  # bool(_seed_travel),

             'loopback': args[i + 15],  # bool(_loopback_mode),
             'prompt_interpolation': args[i + 16],  # bool(_prompt_interpolation),
             'tmpl_pos': args[i + 17],  # str(_tmpl_pos).strip(),
             'tmpl_neg': args[i + 18],  # str(_tmpl_neg).strip(),
             'key_frames': args[i + 19],  # str(_key_frames).strip(),
             'vid_gif': args[i + 20],  # bool(_vid_gif),
             'vid_mp4': args[i + 21],  # bool(_vid_mp4),
             'vid_webm': args[i + 22],  # bool(_vid_webm),
             '_style_pos': args[i + 23],  # str(_style_pos).strip(),
             '_style_neg': args[i + 24],  # str(_style_neg).strip(),
             'source': "",
             'debug': os.path.exists('debug.txt')}

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

    # Save the parameters to a file.
    settings_filename = os.path.join(myset['output_path'], "settings.txt")
    with open(settings_filename, "w+", encoding="utf-8") as f:
        json.dump(myset, f, ensure_ascii=False, indent=4)

    # Have to add the initial picture later on as it doesn't serialise well.
    myset['initial_img'] = args[i + 14]  # _initial_img

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

    # shared.opts.live_previews_enable = tmp_live_previews_enable

    return result, "done"


def ui_block_generation():
    with gr.Blocks():
        with gr.Accordion("Generation Parameters", open=False):
            gr.HTML("<p>These parameters mirror those in txt2img and img2img mode. They are used "
                    "to create the initial image in loopback mode.<br>"
                    "<b>Seed Travel</b>: Allow use of sub seeds to 'smoothly' change from one seed to the next. Only "
                    "makes sense to use if you manually have some seeds set in the keyframes."
                    "</p>")
        steps = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=20)
        from modules.sd_samplers import samplers_for_img2img
        sampler_index = gr.Radio(label='Sampling method',
                                 choices=[x.name for x in samplers_for_img2img],
                                 value=samplers_for_img2img[0].name, type="index")

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
            with gr.Accordion("Initial Image", open=False):
                initial_img = gr.inputs.Image(label='Upload starting image',
                                              image_mode='RGB',
                                              type='pil',
                                              optional=True)

    return steps, sampler_index, width, height, cfg_scale, denoising_strength, seed, seed_travel, initial_img


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
                    steps, sampler_index, width, height, cfg_scale, denoising_strength, seed, seed_travel, image_list =\
                        ui_block_generation()

                    total_time, fps, smoothing, film_interpolation, add_noise, noise_strength, loopback_mode =\
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
                       inputs=[aa_htmllog, steps, sampler_index, width, height, cfg_scale, denoising_strength,
                               total_time, fps, smoothing, film_interpolation, add_noise, noise_strength, seed,
                               seed_travel, image_list, loopback_mode, prompt_interpolation,
                               tmpl_pos, tmpl_neg, key_frames, vid_gif, vid_mp4, vid_webm, style_pos, style_neg],
                       outputs=[aa_gallery, aa_htmlinfo])

        btn_stop.click(fn=lambda: shared.state.interrupt())#,
                       #_js="reenable_animator")

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
