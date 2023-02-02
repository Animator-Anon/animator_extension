from typing import Tuple

from modules import processing, shared, sd_samplers


def setup_processors(mysettings: dict) -> Tuple[processing.StableDiffusionProcessingTxt2Img,
                                                processing.StableDiffusionProcessingImg2Img]:

    ptxt = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
        prompt="",
        styles=['', ''],
        negative_prompt="",
        seed=-1,
        subseed=0,
        subseed_strength=0,
        seed_resize_from_h=None,
        seed_resize_from_w=None,
        seed_enable_extras=False,
        sampler_name=sd_samplers.samplers[mysettings['sampler_index']].name,
        batch_size=1,
        n_iter=1,
        steps=mysettings['steps'],
        cfg_scale=mysettings['cfg_scale'],
        width=mysettings['width'],
        height=mysettings['height'],
        restore_faces=False,
        tiling=False,
        enable_hr=False,
        denoising_strength=mysettings['denoising_strength'],
        firstphase_width=0,
        firstphase_height=0,
        override_settings={},
        do_not_save_samples=True,
        do_not_save_grid=True)

    pimg = processing.StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_img2img_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_img2img_grids,
        prompt='',
        negative_prompt='',
        styles=['', ''],
        seed=-1,
        subseed=0,
        subseed_strength=0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        seed_enable_extras=0,
        sampler_name=sd_samplers.samplers_for_img2img[mysettings['sampler_index']].name,
        batch_size=1,
        n_iter=1,
        steps=mysettings['steps'],
        cfg_scale=mysettings['cfg_scale'],
        width=mysettings['width'],
        height=mysettings['height'],
        restore_faces=False,
        tiling=False,
        init_images=None,
        mask=None,
        mask_blur=0,
        inpainting_fill=0,
        resize_mode=0,
        denoising_strength=mysettings['denoising_strength'],
        inpaint_full_res=False,
        inpaint_full_res_padding=False,
        inpainting_mask_invert=False,
        do_not_save_samples=True,
        do_not_save_grid=True,
    )

    return ptxt, pimg
