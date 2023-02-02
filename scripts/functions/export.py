import os
import subprocess
import glob
from modules import shared


def make_batch_files(my_set: dict):
    final_fps = my_set['fps'] + my_set['fps'] * my_set['smoothing']
    make_gif(my_set['output_path'], 'video', final_fps, False, True)
    make_mp4(my_set['output_path'], 'video', final_fps, False, True)
    make_webm(my_set['output_path'], 'video', final_fps, False, True)


def make_videos(my_set: dict):
    if my_set['film_interpolation']:
        film_interpolation(my_set)
        final_fps = my_set['fps']
        for i in range(my_set['smoothing']):
            final_fps += final_fps - 1
    else:
        final_fps = my_set['fps'] + my_set['fps'] * my_set['smoothing']
    make_gif(my_set['output_path'], 'video', final_fps, my_set['vid_gif'], False)
    make_mp4(my_set['output_path'], 'video', final_fps, my_set['vid_mp4'], False)
    make_webm(my_set['output_path'], 'video', final_fps, my_set['vid_webm'], False)


def film_interpolation(my_set: dict):
    # Need to do a bunch of stuff to copy the frames to the film folder, run that script and then copy them back.
    # Check if FILM exists ...

    film_executable = os.path.basename(shared.opts.animatoranon_film_folder.strip())
    film_folder = os.path.dirname(shared.opts.animatoranon_film_folder.strip())

    if len(film_folder) == 0:
        print('No FILM folder set in options.')
        return

    if not os.path.exists(film_folder):
        print(f'FILM could not be found in this folder: {film_folder}')
        return

    tmp_path = os.path.join(film_folder, 'predict.py')
    if not os.path.exists(tmp_path):
        print(f'FILM could not be found in this folder: {tmp_path}')
        return

    args = [film_executable,
            str(my_set['output_path']),
            str(my_set['smoothing']),
            ]
    subprocess.call(args, cwd=film_folder, shell=True)

    # check it actually worked.
    if not os.path.exists(os.path.join(my_set['output_path'], 'interpolated_frames')):
        print('FILM failed to produce a result.')
        return

    # Delete the files
    filenames = glob.glob(os.path.join(my_set['output_path'], "*.png"))
    for filename in filenames:
        os.remove(filename)

    filenames = glob.glob(os.path.join(my_set['output_path'], 'interpolated_frames', '*.png'))
    i = 0
    for filename in filenames:
        os.rename(filename, os.path.join(my_set['output_path'], f'frame_{i:05d}.png'))
        i += 1


def make_gif(filepath: str, filename: str, fps: float, create_vid: bool, create_bat: bool):
    # Create filenames
    in_filename = f"frame_%05d.png"
    out_filename = f"{str(filename)}.gif"
    # Build cmd for bat output, local file refs only
    cmd = [
        'ffmpeg',
        '-y',
        '-r', str(fps),
        '-i', in_filename.replace("%", "%%"),
        out_filename
    ]
    # create bat file
    if create_bat:
        with open(os.path.join(filepath, "makegif.bat"), "w+", encoding="utf-8") as f:
            f.writelines([" ".join(cmd)])
            # f.writelines([" ".join(cmd), "\r\n", "pause"])
    # Fix paths for normal output
    cmd[5] = os.path.join(filepath, in_filename)
    cmd[6] = os.path.join(filepath, out_filename)
    # create output if requested
    if create_vid:
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def make_webm(filepath: str, filename: str, fps: float, create_vid: bool, create_bat: bool):
    in_filename = f"frame_%05d.png"
    out_filename = f"{str(filename)}.webm"

    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', in_filename.replace("%", "%%"),
        '-crf', str(50),
        '-preset', 'veryfast',
        out_filename
    ]

    if create_bat:
        with open(os.path.join(filepath, "makewebm.bat"), "w+", encoding="utf-8") as f:
            f.writelines([" ".join(cmd)])
            # f.writelines([" ".join(cmd), "\r\n", "pause"])

    cmd[5] = os.path.join(filepath, in_filename)
    cmd[10] = os.path.join(filepath, out_filename)

    if create_vid:
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def make_mp4(filepath: str, filename: str, fps: float, create_vid: bool, create_bat: bool):
    in_filename = f"frame_%05d.png"
    out_filename = f"{str(filename)}.mp4"

    cmd = [
        'ffmpeg',
        '-y',
        '-r', str(fps),
        '-i', in_filename.replace("%", "%%"),
        '-c:v', 'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'veryfast',
        out_filename
    ]

    if create_bat:
        with open(os.path.join(filepath, "makemp4.bat"), "w+", encoding="utf-8") as f:
            f.writelines([" ".join(cmd)])
            # f.writelines([" ".join(cmd), "\r\n", "pause"])

    cmd[5] = os.path.join(filepath, in_filename)
    cmd[16] = os.path.join(filepath, out_filename)

    if create_vid:
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
