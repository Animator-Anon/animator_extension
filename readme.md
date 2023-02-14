# Animation Script<a name="animationscript"></a>
A basic img2img script that will dump frames and build a video file. Suitable for creating interesting
zoom-in warping movies, but not too much else at this time. The basic idea is to story board some kind 
of animation with changes in prompts, translation etc at a low framerat until you get it roughly right.
Then bump up the framerate for a final render, it should play out roughly the same, just  more detail.

This is intended to be a versatile toolset to help you automate some img2img tasks. It certainly isn't
a one-stop shop for digital movies.

Inspired by Deforum Notebook, which has not turned into an extension. This might be a bit easier to use but is probably not as powerful.
Must have ffmpeg installed in path to create movies, but image sequences will be created regardless. 
This suffers from img2img embossing, if the image is static for too long. I would have to look at 
someone else's implementation to figure out why and don't want to steal their code.

## Table of Contents
1. [Animation Script](#animationscript)
2. [Major Features](#majorfeatures)
    1. [FILM](#film)
3. [Installation](#installation)
4. [Example](#example)
5. [Settings](#settings)
    1. [persistent_settings](#persistent_settings)
    2. [video_formats](#video_formats)
    3. [total_animation_length](#total_animation_length)
    4. [framerate](#framerate)
    5. [denoising_strength](#denoising_strength)
    6. [denoising_decay](#denoising_decay)
    7. [zoom_factor](#zoom_factor)
    8. [x_pixel_shift](#x_pixel_shift)
    9. [y_pixel_shift](#y_pixel_shift)
    10. [templates](#templates)
6. [Keyframes](#keyframes)
    1. [clear_stamp](#clear_stamp)
    2. [clear_text](#clear_text)
    3. [denoise](#denoise)
    4. [model](#model)
    5. [prompt](#prompt)
    6. [prompt_from_png](#prompt_from_png)
    7. [prompt_vtt](#prompt_vtt)
    8. [prop](#prop)
    9. [seed](#seed)
    10. [set_stamp](#set_stamp)
    11. [set_text](#set_text)
    12. [source](#source)
    13. [template](#template)
    14. [transform](#transform)
7. [Changelog](#changelog)

# Major Features<a name="majorfeatures"></a>
- Transformation
    - In img2img mode, the picture can be zoomed and panned around to create pull shots, pans and the 
      like. Off-frame sections will be filled in with edge pixels.
- Prompt marching
    - Prompts will be walked from one to the next. The Both prompts will be AND together with opposing
      weights.
- Seed marching
    - In txt2img mode, a seed value can be walked to another using subseed strength. It won't be a 
      seamless transition. Parts of the image will change independently in sequence.
- Text boxes, props
    - Post-processing effects can be added onto frames that are written to disk. Maybe used for 
      creating interesting videos.

## Installation<a name="installation"></a>
To create videos, you need FFMPEG installed and available in your path. i.e. open a command line and type in ffmpeg.exe,
you should be able to run the program.<br>

This extension can be installed from the extension tab in webUI. Enter the GitHub link in the 
installation tab on the extensions tab.

`https://github.com/Animator-Anon/animator_extension`

Or use this command:

`git clone https://github.com/Animator-Anon/animator_extension extensions/animator_extension`

### FILM<a name="film"></a>
FILM needs to be installed separately, it is not bundled with this extension. Head to 
https://github.com/google-research/frame-interpolation to get it done. Once you have it installed, create a .bat file 
that takes two arguments and can run the script from any location. The first argument is the source picture location, 
which this script will set automatically, and the second is the number of iterations, or sub frames to add.
In my case, I installed FILM in a python VENV, different to the one the webUI is running in, so I have to change to it:<br>
```
call %USERPROFILE%\miniconda3\condabin\conda.bat activate ldm
python -m eval.interpolator_cli --pattern "%1" --model_path pretrained_models\film_net\Style\saved_model --times_to_interpolate %2
call %USERPROFILE%\miniconda3\condabin\conda.bat deactivate
```

## Keyframe example:<a name="example"></a>
There seems to be confusion on how to use the keyframes. Copy them into the keyframe box and change the values to suit.
Here is an example:

    0 | template | black and white, silent film, film grain | low quality, cropped
    0 | transform | 2 | 0 | 0 | 0
    0   | prompt | Shot of a sleeping person, possibly the narrator, in a dreamlike state | 
    8   | prompt | Shot of a bright sun being extinguished, symbolizing the loss of hope |
    16  | prompt | Close-up shot of stars wandering aimlessly in the darkness of space |
    24  | prompt | Close-up shot of the barren and desolate landscape, with no sign of life or movement |
    32  | prompt | Wide shot of the icy and dark earth, spinning in the moonless air |

This will set up at time 0, the templates which will be appended onto the prompts at each prompt keyframe.
Also a transform is started at time 0. The effect of all of these will be a zoom in on a constantly changing scene, as
the different prompts are applied at the times. No seeds have been specified, so they will be random.

## Settings:<a name="settings"></a>
Many explanations exist in the up in expandable sections of the page. Look for a triangle right side.

### Persistent Settings:<a name="persistent_settings"></a>
There are some persistent settings that are on the WebUI settings page. These will be stored by WebUI.
- FILM batch or script file, including full path
    - Full path to a batch file that can be called for FILM interpolation.
- Prop folder
    - The folder that prop pictures will be read from.
- New output folder
    - Output folder specifically used by this extension. Saves loading up the general output folders. 

### Video formats:<a name="video_formats"></a>
Create GIF, webM or MP4 file from the series of images. Regardless, .bat files will be created with the right options to
make the videos at a later time.

### Total Animation Length (s):<a name="total_animation_length"></a>
Total number of seconds to create. Will create fps frames for every second, as you'd expect.

### Framerate:<a name="framerate"></a>
Frames per second to generate.

### Denoising Strength:<a name="denoising_strength"></a>
Initial denoising strength value, overrides the value above which is a bit strong for a default. Will be overridden by
keyframes when they are hit.
Note that denoising is not scaled by fps, like other parameters are.

### Denoising Decay:<a name="denoising_decay"></a>
Experimental option to enable a half-life decay on the denoising strength. Its value is halved every second. Not that
useful because of img2img embossing.

### Zoom Factor (scale/s):<a name="zoom_factor"></a>
Initial zoom in (>1) or out (<1), at this rate per second. E.g. 2.0 will double size (and crop) every second. Will be
overridden by keyframes when they are hit.

### X Pixel Shift (pixels/s):<a name="x_pixel_shift"></a>
Shift the image right (+) or left (-) in pixels per second. Will be overridden by keyframes when they are hit.

### Y Pixel Shift (pixels/s):<a name="y_pixel_shift"></a>
Shift the image down (+) or up (-) in pixels per second. Will be overridden by keyframes when they are hit.

### Templates:<a name="templates"></a>
Provide common positive and negative prompts for each keyframe below, save typing them out over and over. They will only
be applied when a keyframe is hit. The prompts in the keyframes will be appended to these and sent for processing until
the next keyframe that has a prompt.

### Keyframes:<a name="keyframes"></a>
Key frames have been broken down into individual commands, since the old keyframe was blowing out.
Commands:

### source<a name="source"></a>
Set source of frames for processing.

- Format: `time_s | source | video, images, img2img | path`
- time_s: Time in seconds from the start to make the change.
- prompt: video, images, img2img. Source for the video frames. Default img2img.
- path: Either the file name of the video file, or the path and wildcard filename of the images.

### prompt<a name="prompt"></a>
Set positive and negative prompts.

- Format: `time_s | prompt | positive_prompts | negative_prompts`
- time_s: Time in seconds from the start to make the change.
- prompt: Command name.
- positive_prompts: Replacement positive prompts. Will be concatenated with the positive template.
- negative_prompts: Replacement negative prompts. Will be concatenated with the negative template.

### template<a name="template"></a>
Set positive and negative prompt template. Purely saves filling out the boxes above in the web UI.

- Format: `time_s | prompt | positive_prompts | negative_prompts`
- time_s: Time in seconds from the start to make the change.
- template: Command name.
- positive_prompts: Replacement positive prompts. Will be concatenated with the positive template.
- negative_prompts: Replacement negative prompts. Will be concatenated with the negative template.

### prompt_from_png<a name="prompt_from_png"></a>
Sets the seed, positive and negative prompts from the specified png file, if it contains it.

- Format: `time_s | prompt_from_png | file_path`
- time_s: Time in seconds from the start to make the change.
- prompt_from_png: Command name.
- file path and name to a png file that contains the info you want.

### prompt_vtt<a name="prompt_vtt"></a>
Loads a series of prompts from a .vtt subtitle file. The first cue is read as positive prompts | negative prompts, and
set at the specified cue time.

- Format: `time_s | prompt_vtt | vtt_filepath`
- time_s: Time in seconds from the start to make the change.
- prompt_vtt: Command name.
- file path and name to a vtt file that contains the positive and negative prompts.

### transform<a name="transform"></a>
Set the current transform.

- Format: `time_s | transform | zoom | x_shift | y_shift | rotation`
- time_s: Time in seconds from the start to make the change.
- transform: Command name.
- zoom: New zoom value. 1 = 100% per second. 2 = zoom in 200% over 1 second.
- x_shift: X shift value, in pixels per second.
- y_shift: Y shift value, in pixels per second.
- rotation: Rotation, in degrees per second.

### seed<a name="seed"></a>
Force a specific seed. It's technically a thing you can do, how usefull it is, is up to you to decide.

- Format: `time_s | seed | new_seed_int`
- time_s: Time in seconds from the start to make the change.
- denoise: Command name.
- denoise_value: New denoise strength value.

### denoise<a name="denoise"></a>
Set the denoise strength.

- Format: `time_s | denoise | denoise_value`
- time_s: Time in seconds from the start to make the change.
- denoise: Command name.
- denoise_value: New denoise strength value.

### set_text<a name="set_text"></a>
Overlay a rounded text box in post-processing. I.e. only applied to the image that is saved, and it is not iterated on.
The text will be centered in the box with the largest font size that will fit.
Text boxes are referenced by the name you give. If you set it again, you can change the contents. Or it can be cleared.
Multiple text boxes with different names can exist at the same time.

- Format:`time_s | set_text | textblock_name | text_prompt | x | y | w | h | fore_color | back_color | font_name`
- time_s: Time in seconds from the start to make the change.
- set_text: Command name.
- textblock_name: Unique name or tag given to this text block in the set_text command above.
- text_prompt: Text to put in the text block. You can use \n for multi-line.
- x: Top left X position of the text block.
- y: Top left Y position of the text block.
- w: Width of text block
- h: Height of text block
- fore_color: colour name as string, or a tuple of bytes (127,0,127)
- back_color: colour name as string, or a tuple of bytes (127,0,127)
- font_name: name of the font file. Python will attempt to scan your system font folders for this file.

### clear_text<a name="clear_text"></a>
Remove a named text box, it will no longer be drawn on the saved images.

- Format: `time_s | clear_text | textblock_name`
- time_s: Time in seconds from the start to make the change.
- clear_text: Command name.
- textblock_name: Unique name or tag given to this text block in the set_text command above.

### prop<a name="prop"></a>
Embed a clipart image into the picture to be diffused. it will be drawn once at this time. You need to set a prop folder
where transparent pngs are held, and specify them by file name.

- Format: `time_s | prop | prop_filename | x_pos | y_pos | scale | rotation`
- time_s: Time in seconds from the start to make the change.
- prop: Command name.
- prop_filename: Name of a picture in the props folder, hopefully with transparency.
- x_pos: Center position of the prop.
- y_pos: Center position of the prop.
- scale: Scale value. 1=100% etc.
- rotation: Rotation, in degrees.

### set_stamp<a name="set_stamp"></a>
Like props but applied in post processing and will not be diffused. You ca reference them by name, change their details
on the fly as with text boxes.

- Format: `time_s | set_stamp | stamp_name | stamp_filename | x_pos | y_pos | scale | rotation`
- time_s: Time in seconds from the start to make the change.
- set_stamp: Command name.
- stamp_name: Unique name or tag given to this stamp. Used to change it's parameters or delete.
- stamp_filename: Name of a picture in the props folder, hopefully with transparency.
- x_pos: Center position of the stamp.
- y_pos: Center position of the stamp.
- scale: Scale value. 1=100% etc.
- rotation: Rotation, in degrees.

### clear_stamp<a name="clear_stamp"></a>
Clear out a stamp, will no longer be drawn on the saved images.

- Format: `time_s | clear_stamp | stamp_name`
- time_s: Time in seconds from the start to make the change.
- clear_stamp: Command name.
- stamp_name: Unique name or tag given to this stamp in the set_stamp command above.

### model<a name="model"></a>
Allows you to change the model on the fly, if you need to. It won't change it back at the end, so if you do use this,
maybe set the initial model in frame 0 first.

- Format: `time_s | model | model_name`
- time_s: Time in seconds from the start to make the change.
- model: Command name.
- model_name: Pick one from the list. Just the name with no extension or hash is fine.

### Changelog<a name="changelog"></a>
- Extension
    - Re-fectored script into an extension, easier to manage and work with.
- v6
    - Allowed the script to work in txt2img tab. This is mainly for seed marching, as it doesn't make sense to recycle
      the images in this mode.
    - Each frame is a fresh generation between two seeds.
    - Re-enabled transforms for video and image sources. Requested feature.
    - Added some new prompt commands.
- v5
    - Added some different sources for frames. If video is specified, frames will be grabbed from the supplied video
      file, until the end of the video is hit. Then the last frame will be repeated (i think). Images works silimarly,
      but you need to specify a path and wilcard file name, such as c:/images/*.png. A list of images will be created,
      alphabetically sorted by default. If there are not enough images for the specified duration, the last frame will
      be repeated.
    - Specifying a different source other than img2img means the initial image in the UI will be ignored, but it needs
      to be there for the UI to work.
    - Further, transformation won't have an effect, as there is no longer a feedback loop.
- v4
    - Investigating seed marching. Look at some scripts and figured out how subseeds are meant to be used. Implemented
      it as an option. If you add seed keyframes and check the seed march box, the script will migrate from one seed to
      the next using the subseed & subseed strength interpolation. if you leave seed march unckecked, the seed value
      will interpolate from one integer to the next. View the final values in the dataframe .csv dump in the output
      folder.
    - Seed marching doesn't seem to work well with transforms, if you combine them you'll get poor results.
- v3
    - Changed the interpolation system, so various keyframes write values into a frame dataframe. Blank values are then
      interpolated, so transform changes are smoother. also used to interpolated prompts by default now.
- v2
    - Broken down the growing prompt into individual commands to control various parameters. Added requested features.
      Rolled in changed from pull requests, thanks to contributers.
- Original
    - Original version of the script with one basic prompt format.
