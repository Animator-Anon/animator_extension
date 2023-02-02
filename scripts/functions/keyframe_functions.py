import glob
import math
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import piexif
import piexif.helper
from PIL import Image

from modules import shared


def read_vtt(filepath: str, total_time: float, fps: float) -> list:
    vtt_list = []
    if not os.path.exists(filepath):
        print("VTT: Cannot locate vtt file: " + filepath)
        return vtt_list

    with open(filepath, 'r') as vtt_file:
        tmp_vtt_line = vtt_file.readline()
        tmp_vtt_frame_no = 0
        if "WEBVTT" not in tmp_vtt_line:
            print("VTT: Incorrect header: " + tmp_vtt_line)
            return vtt_list

        while 1:
            tmp_vtt_line = vtt_file.readline()
            if not tmp_vtt_line:
                break

            tmp_vtt_line = tmp_vtt_line.strip()
            if len(tmp_vtt_line) < 1:
                continue

            if '-->' in tmp_vtt_line:
                # 00:00:01.510 --> 00:00:05.300
                tmp_vtt_a = tmp_vtt_line.split('-->')
                # 00:00:01.510
                tmp_vtt_b = tmp_vtt_a[0].split(':')
                if len(tmp_vtt_b) == 2:
                    # [00,05.000]
                    tmp_vtt_frame_time = float(tmp_vtt_b[1]) + \
                                         60.0 * float(tmp_vtt_b[0])
                elif len(tmp_vtt_b) == 3:
                    # [00,00,01.510]
                    tmp_vtt_frame_time = float(tmp_vtt_b[2]) + \
                                         60.0 * float(tmp_vtt_b[1]) + \
                                         3600.0 * float(tmp_vtt_b[0])
                else:
                    # Badly formatted time string. Set high value to skip next prompt.
                    tmp_vtt_frame_time = 1e99
                tmp_vtt_frame_no = int(tmp_vtt_frame_time * fps)

            if '|' in tmp_vtt_line:
                # pos prompt | neg prompt
                tmp_vtt_line_parts = tmp_vtt_line.split('|')
                if len(tmp_vtt_line_parts) >= 2 and tmp_vtt_frame_time < total_time:
                    vtt_list.append((tmp_vtt_frame_no,
                                     tmp_vtt_line_parts[0].strip().lstrip('-').strip(),
                                     tmp_vtt_line_parts[1]))
                    tmp_vtt_frame_time = 1e99

    return vtt_list


def get_pnginfo(filepath: str) -> Tuple[bool, str, str]:

    worked = False

    if not os.path.exists(filepath):
        return worked, '', 'Error: Could not find image.'

    image = Image.open(filepath, "r")

    if image is None:
        return worked, '', 'Error: No image supplied'

    items = image.info
    geninfo = ''

    if "exif" in image.info:
        exif = piexif.load(image.info["exif"])
        exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
        try:
            exif_comment = piexif.helper.UserComment.load(exif_comment)
        except ValueError:
            exif_comment = exif_comment.decode('utf8', errors="ignore")

        items['exif comment'] = exif_comment
        geninfo = exif_comment

        for field in ['jfif', 'jfif_version', 'jfif_unit', 'jfif_density', 'dpi', 'exif',
                      'loop', 'background', 'timestamp', 'duration']:
            items.pop(field, None)

    geninfo = items.get('parameters', geninfo)

    info = ''
    for key, text in items.items():
        info += f"{str(key).strip()}:{str(text).strip()}".strip()+"\n"

    if len(info) == 0:
        info = "Error: Nothing found in the image."
    else:
        worked = True

    return worked, geninfo, info


# Process the keyframe string and build the dataframe of all the changing parameters
def process_keyframes(mysettings: dict) -> pd.DataFrame:

    mysettings['keyframes'] = {}      # Dict of keyframes, where the index will be the frame it takes effect.
    my_prompts = []     # List of tuple of prompts
    my_seeds = {}       # dict of seeds

    frame_count = math.ceil(mysettings['fps'] * mysettings['total_time'])

    # Try and apply styles in addition to what was written into the template text boxes.
    mysettings['tmpl_pos'] = shared.prompt_styles.apply_styles_to_prompt(mysettings['tmpl_pos'],
                                                                         [mysettings['_style_pos'], 'None'])
    mysettings['tmpl_neg'] = shared.prompt_styles.apply_negative_styles_to_prompt(mysettings['tmpl_neg'],
                                                                                  [mysettings['_style_neg'], 'None'])

    # Define the columns in the pandas dataframe that will hold and calculate all the changing values..
    variables = {'pos1': np.nan,
                 'neg1': np.nan,
                 'pos2': np.nan,
                 'neg2': np.nan,
                 'prompt': np.nan,
                 'denoise': np.nan,
                 'noise': np.nan,
                 'x_shift': np.nan,
                 'y_shift': np.nan,
                 'zoom': np.nan,
                 'rotation': np.nan,
                 'cfg_scale': np.nan}

    # Create the dataframe
    df = pd.DataFrame(variables, index=range(frame_count + 1))

    # Preload the dataframe with some values, so they can be filled down correctly.
    df.loc[0, ['denoise', 'x_shift', 'y_shift', 'zoom', 'rotation', 'noise', 'cfg_scale']] = \
        [mysettings['denoising_strength'],
         0.0, 0.0, 1.0, 0.0,
         mysettings['noise_strength'],
         mysettings['cfg_scale']]

    # Iterate through the supplied keyframes, splitting by newline.
    for key_frame in mysettings['key_frames'].splitlines():

        # Ignore comments
        if key_frame.strip().startswith("#"):
            continue

        # Break keyframe into sections.
        key_frame_parts = key_frame.split("|")
        if len(key_frame_parts) < 2:
            continue

        # Figure out frame number and command
        tmp_frame_no = int(float(key_frame_parts[0]) * mysettings['fps'])
        tmp_command = key_frame_parts[1].lower().strip()

        if tmp_frame_no not in mysettings['keyframes']:
            mysettings['keyframes'][tmp_frame_no] = []
        mysettings['keyframes'][tmp_frame_no].append(key_frame_parts[1:])

        # Switch on command and load in the appropriate data into the dataframe
        if tmp_command == "transform" and len(key_frame_parts) == 6:
            # Time (s) | transform  | Zoom (/s) | X Shift (pix/s) | Y shift (pix/s) | Rotation (deg/s)
            df.loc[tmp_frame_no,
                   ['x_shift', 'y_shift', 'zoom', 'rotation']
                   ] = [float(key_frame_parts[3]) / mysettings['fps'],
                        float(key_frame_parts[4]) / mysettings['fps'],
                        float(key_frame_parts[2]) ** (1.0 / mysettings['fps']),
                        float(key_frame_parts[5]) / mysettings['fps']]
        elif tmp_command == "denoise" and len(key_frame_parts) == 3:
            # Time (s) | denoise | denoise
            df.loc[tmp_frame_no, ['denoise']] = [float(key_frame_parts[2])]
        elif tmp_command == "cfg_scale" and len(key_frame_parts) == 3:
            # Time (s) | cfg_scale | cfg_scale
            df.loc[tmp_frame_no, ['cfg_scale']] = [float(key_frame_parts[2])]
        elif tmp_command == "noise" and len(key_frame_parts) == 3:
            # Time (s) | noise | noise_strength
            df.loc[tmp_frame_no, ['noise']] = [float(key_frame_parts[2])]
        elif tmp_command == "seed" and len(key_frame_parts) == 3:
            # Time (s) | seed | seed
            my_seeds[tmp_frame_no] = int(key_frame_parts[2])
        elif tmp_command == "prompt" and len(key_frame_parts) >= 3:
            # Time (s) | prompt | Positive Prompts | Negative Prompts
            if len(key_frame_parts) == 4:
                my_prompts.append((tmp_frame_no, key_frame_parts[2].strip().strip(",").strip(),
                                   key_frame_parts[3].strip().strip(",").strip()))
            else:
                # no negative prompt supplied.
                my_prompts.append((tmp_frame_no, key_frame_parts[2].strip().strip(",").strip(), ''))
        elif tmp_command == "prompt_vtt" and len(key_frame_parts) == 3:
            # Time (s) | prompt_vtt | vtt_filepath
            vtt_prompts = read_vtt(key_frame_parts[2].strip(), mysettings['total_time'], mysettings['fps'])
            for vtt_time, vtt_pos, vtt_neg in vtt_prompts:
                my_prompts.append((vtt_time, vtt_pos.strip().strip(",").strip(),
                                   vtt_neg.strip().strip(",").strip()))
        elif tmp_command == "template" and len(key_frame_parts) == 4:
            # Time (s) | template | Positive Prompts | Negative Prompts
            mysettings['tmpl_pos'] = key_frame_parts[2].strip().strip(",").strip()
            mysettings['tmpl_neg'] = key_frame_parts[3].strip().strip(",").strip()
        elif tmp_command == "prompt_from_png" and len(key_frame_parts) == 3:
            # Time (s) | prompt_from_png | file name
            tmp_png_filename = key_frame_parts[2].strip().strip(",").strip()
            foundinfo, geninfo, info = get_pnginfo(tmp_png_filename)
            if not foundinfo:
                print(f"Error with PNG: {tmp_png_filename}: {info}")
                # print(geninfo)
            else:
                # print(geninfo)
                if "\nNegative prompt:" in geninfo:
                    # print("DBG: found pos + neg")
                    tmp_posprompt = geninfo[:geninfo.find("\nNegative prompt:")]
                    tmp_negprompt = geninfo[geninfo.find("\nNegative prompt:")+18:geninfo.rfind("\nSteps:")]
                else:
                    # print("DBG: found pos")
                    tmp_posprompt = geninfo[:geninfo.find("\nSteps:")]
                    tmp_negprompt = ''

                tmp_params = geninfo[geninfo.rfind("\nSteps:")+1:]
                tmp_seed = int(tmp_params[tmp_params.find('Seed: ') + 6:
                                          tmp_params.find(",", tmp_params.find('Seed: ') + 6)])
                # print(f"Pos:[{tmp_posprompt}] Neg:[{tmp_negprompt}] Seed:[{tmp_seed}]")
                my_prompts.append((tmp_frame_no, tmp_posprompt, tmp_negprompt))
                my_seeds[tmp_frame_no] = tmp_seed
        elif tmp_command == "source" and len(key_frame_parts) > 2:
            # time_s | source | source_name | path
            tmp_source_name = key_frame_parts[2].lower().strip()
            tmp_source_path = key_frame_parts[3].lower().strip()
            if tmp_source_name == 'video':
                if os.path.exists(tmp_source_path):
                    mysettings['source'] = tmp_source_name
                    mysettings['source_file'] = tmp_source_path
                else:
                    print(f"Could not locate video: {tmp_source_path}")
            elif tmp_source_name == 'images':
                source_cap = glob.glob(tmp_source_path)
                if len(source_cap) > 0:
                    mysettings['source'] = tmp_source_name
                    mysettings['source_file'] = source_cap
                    print(f'Found {len(source_cap)} images in {tmp_source_path}')
                else:
                    print(f'No images found, reverting back to img2img: {tmp_source_path}')

    # Sort the dict of prompts by frame number, and then populate the dataframe in a alternating fashion.
    # need to do this to ensure the prompts flow onto each other correctly.
    my_prompts = sorted(my_prompts)

    # Special case if no prompts supplied.
    if len(my_prompts) == 0:
        df.loc[0, ['pos1', 'neg1', 'pos2', 'neg2', 'prompt']] = ["", "", "", "", 1.0]
    elif len(my_prompts) == 1:
        df.loc[0, ['pos1', 'neg1', 'pos2', 'neg2', 'prompt']] = [my_prompts[0][1], my_prompts[0][2], "", "", 1.0]
    else:
        for x in range(len(my_prompts) - 1):
            df.loc[my_prompts[x][0], ['pos1', 'neg1', 'pos2', 'neg2', 'prompt']] = [my_prompts[x][1],
                                                                                    my_prompts[x][2],
                                                                                    my_prompts[x + 1][1],
                                                                                    my_prompts[x + 1][2],
                                                                                    1]
            if x > 0:
                df.loc[my_prompts[x][0] - 1, 'prompt'] = 0

    df.at[df.index[-1], 'prompt'] = 0
    df.loc[:, ['pos1', 'neg1', 'pos2', 'neg2']] = df.loc[:, ['pos1', 'neg1', 'pos2', 'neg2']].ffill()

    # Fill out the seeds
    for x in my_seeds:
        if my_seeds[x] == -1:
            my_seeds[x] = int(random.randrange(4294967294))

    if 0 not in my_seeds:
        print("DBG seed: No initial seed provided, adding UI one.")
        if mysettings['seed'] == -1:
            mysettings['seed'] = int(random.randrange(4294967294))
            print("DBG seed: Generating random seed.")
        my_seeds[0] = mysettings['seed']

    print("DBG seed: Sorting list of seeds:")
    print(my_seeds)
    print("DBG seed: List of prompts:")
    print(my_prompts)

    if len(my_seeds) > 1:
        # Seed commands given.
        if mysettings['seed_travel']:
            # print("DBG seed: More than 1 seed, seed travel enabled.")
            # Try to interpolate from seed -> sub-seed, by increasing sub-seed strength
            idxs = list(my_seeds.keys())
            idxs.sort()
            for idx in range(len(idxs)-1):
                df.loc[idxs[idx], ['seed_start', 'seed_end', 'seed_str']] = [str(my_seeds[idxs[idx]]),
                                                                             str(my_seeds[idxs[idx + 1]]),
                                                                             0]
                if idx == len(my_seeds) - 2:
                    df.at[df.index[-1], 'seed_str'] = 1
                if idx > 0:
                    df.loc[idxs[idx] - 1, 'seed_str'] = 1  # Ensure all values tend to one in the list
            # print(df[['seed_start', 'seed_end', 'seed_str']])
            df.loc[:, ['seed_start', 'seed_end']] = df.loc[:, ['seed_start', 'seed_end']].ffill()
            # print(df[['seed_start', 'seed_end', 'seed_str']])
        else:
            # print("DBG seed: More than 1 seed, seed travel disabled.")
            # Just interpolate from one seed value to the next. experimental. Set sub-seed to None to disable.
            for idx in my_seeds:
                df.loc[idx, 'seed_start'] = my_seeds[idx]
            # print(df['seed_start'])
            df.loc[:, 'seed_start'] = df.loc[:, 'seed_start'].interpolate(limit_direction='both').map(int)
            # print(df['seed_start'])
            df['seed_end'] = None
            df['seed_str'] = 0
    else:
        # print("DBG seed: Only one seed, series fill.")
        # Only initial seed given, load in initial value, series fill. Set sub-seed to None to disable.
        df.at[0, 'seed_start'] = my_seeds[0]
        df.at[df.index[-1], 'seed_start'] = my_seeds[0] + frame_count
        df.loc[:, 'seed_start'] = df.loc[:, 'seed_start'].interpolate(limit_direction='both').map(int)
        df['seed_end'] = None
        df['seed_str'] = 0

    # Interpolate columns individually depending on how many data points.
    for name, values in df.items():
        if name in ['prompt', 'seed_str']:
            df.loc[:, name] = df.loc[:, name].interpolate(limit_direction='both')
        elif values.count() > 3:
            df.loc[:, name] = df.loc[:, name].interpolate(limit_direction='both', method="polynomial", order=2)
            df.loc[:, name] = df.loc[:, name].interpolate(limit_direction='both')  # catch last null values.
        else:
            df.loc[:, name] = df.loc[:, name].interpolate(limit_direction='both')

    if mysettings['prompt_interpolation']:
        # Check if templates are filled in. If not, try grab prompts at top (i.e. image sent from png info)
        if len(mysettings['tmpl_pos']) == 0:
            df['pos_prompt'] = df['pos1'].map(str) + ':' + df['prompt'].map(str) + ' AND ' + \
                               df['pos2'].map(str) + ':' + (1.0 - df['prompt']).map(str)
        else:
            df['pos_prompt'] = mysettings['tmpl_pos'] + ',' + df['pos1'].map(str) + ':' + df['prompt'].map(str) +\
                               ' AND ' + mysettings['tmpl_pos'] + ',' + df['pos2'].map(str) + ':' + \
                               (1.0 - df['prompt']).map(str)
        if len(mysettings['tmpl_neg']) == 0:
            df['neg_prompt'] = df['neg1'].map(str) + ':' + df['prompt'].map(str) + ' AND ' + \
                               df['neg2'].map(str) + ':' + (1.0 - df['prompt']).map(str)
        else:
            df['neg_prompt'] = mysettings['tmpl_neg'] + ',' + df['neg1'].map(str) + ':' + df['prompt'].map(str) + \
                               ' AND ' + mysettings['tmpl_neg'] + ',' + df['neg2'].map(str) + ':' + \
                               (1.0 - df['prompt']).map(str)
    else:
        if len(mysettings['tmpl_pos']) == 0:
            df['pos_prompt'] = df['pos1'].map(str)
        else:
            df['pos_prompt'] = mysettings['tmpl_pos'] + ',' + df['pos1'].map(str)
        if len(mysettings['tmpl_neg']) == 0:
            df['neg_prompt'] = df['neg1'].map(str)
        else:
            df['neg_prompt'] = mysettings['tmpl_neg'] + ',' + df['neg1'].map(str)

    csv_filename = os.path.join(mysettings['output_path'], "keyframes.csv")
    df.to_csv(csv_filename)

    return df
