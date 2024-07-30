import json
cfg = None

# -------------------------------------------------------------------------------------------
with open("config/Stable Diffusion 1.5.json", 'r') as f:
    cfg = json.load(f)
# -------------------------------------------------------------------------------------------
assert cfg is not None, "cfg is None."
import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg['gpu']}"
import shutil
import cv2
import torch
import einops
import numpy as np
# from datetime import datetime
from PIL import Image
from pytorch_lightning import seed_everything
import torchvision.transforms as transformers

from enum import Enum
from ControlNet.cldm.model import create_model, load_state_dict
from ControlNet.annotator.canny import CannyDetector
from ControlNet.annotator.hed import HEDdetector
from ControlNet.annotator.midas import MidasDetector
from ControlNet.annotator.util import HWC3
from safetensors.torch import load_file

from src.controller import AttentionControl
from src.ddim_v_hacked import DDIMVSampler
from src.img_util import numpy2tensor

import huggingface_hub

device = "cuda" if torch.cuda.is_available() else "cpu"

# use_control = False


steps = cfg['steps']
sd_model = cfg['sd_model']
# cur_time = datetime.now().strftime("%Y%m%d_%H%M")
x0_strength = cfg['x0_strength']
use_control = cfg['use_control']
control_type = 'canny' if use_control else 'nocontrol'
# -------------------------------------------------------------------------------------------
feature_path = os.path.join(cfg['feature_path'], device+f"{cfg['gpu']}_{control_type}_{x0_strength}_coco")
# feature_path = os.path.join(cfg['feature_path'], device+f"{cfg['gpu']}_{control_type}_{x0_strength}_imagenet-r")
# feature_path = os.path.join(cfg['feature_path'], device+f"{cfg['gpu']}_{control_type}_{x0_strength}_imagenet")
# -------------------------------------------------------------------------------------------
prompt = cfg['prompt']
a_prompt = cfg['a_prompt']
n_prompt = cfg['n_prompt']

low_threshold = 50
high_threshold = 100

print(device)
print(steps)
print(sd_model)
print(use_control)
print(feature_path)
print(prompt)
print(a_prompt)
print(n_prompt)

eta = 0.0

model_dict = {
    'Stable Diffusion 1.5': '/home/yfyuan/YYF/all_models/model.ckpt',
    'revAnimated_v11': 'models/revAnimated_v11.safetensors',
    'realisticVisionV20_v20': 'models/realisticVisionV20_v20.safetensors',
    'DGSpitzer/Cyberpunk-Anime-Diffusion': '~/YYF/all_models/Cyberpunk-Anime-Diffusion.safetensors',
    'wavymulder/Analog-Diffusion': 'analog-diffusion-1.0.safetensors',
    'Stable_Diffusion_PaperCut_Model': '/home/yfyuan/YYF/all_models/papercut_v1.ckpt',
}

class ProcessingState(Enum):
    NULL = 0
    FIRST_IMG = 1
    KEY_IMGS = 2


class GlobalState:

    def __init__(self):
        self.sd_model = None
        self.ddim_v_sampler = None
        self.detector_type = None
        self.detector = None
        self.controller = None
        self.processing_state = ProcessingState.NULL

    def update_controller(self, inner_strength, mask_period, cross_period,
                          ada_period, warp_period):
        self.controller = AttentionControl(inner_strength, mask_period,
                                           cross_period, ada_period,
                                           warp_period)

    def update_sd_model(self, sd_model, control_type):
        if sd_model == self.sd_model:
            return
        self.sd_model = sd_model
        model = create_model('./ControlNet/models/cldm_v15.yaml').cpu()
        if control_type == 'HED':
            # model.load_state_dict(
            #     load_state_dict(huggingface_hub.hf_hub_download(
            #         'lllyasviel/ControlNet', './models/control_sd15_hed.pth'),
            #         location=device))
            model.load_state_dict(
                load_state_dict('./models/control_sd15_hed.pth',
                                location=device))
        elif control_type == 'canny':
            model.load_state_dict(
                load_state_dict('./models/control_sd15_canny.pth',
                                location=device))
        elif control_type == 'depth':
            model.load_state_dict(
                load_state_dict('./models/control_sd15_depth.pth',
                                location=device))

        model.to(device)
        sd_model_path = model_dict[sd_model]
        if len(sd_model_path) > 0:
            # check if sd_model is repo_id/name otherwise use global REPO_NAME
            if sd_model.count('/') == 1:
                repo_name = sd_model

            model_ext = os.path.splitext(sd_model_path)[1]
            if model_ext == '.safetensors':
                model.load_state_dict(load_file(sd_model_path), strict=False)
            elif model_ext == '.ckpt' or model_ext == '.pth':
                model.load_state_dict(torch.load(sd_model_path)['state_dict'],
                                      strict=False)

        try:
            model.first_stage_model.load_state_dict(torch.load('./models/vae-ft-mse-840000-ema-pruned.ckpt')['state_dict'],strict=False)
        except Exception:
            print('Warning: We suggest you download the fine-tuned VAE',
                  'otherwise the generation quality will be degraded')

        self.ddim_v_sampler = DDIMVSampler(model)

    def clear_sd_model(self):
        self.sd_model = None
        self.ddim_v_sampler = None
        if device == 'cuda':
            torch.cuda.empty_cache()

    def update_detector(self, control_type, canny_low=100, canny_high=200):
        if self.detector_type == control_type:
            return
        if control_type == 'HED':
            self.detector = HEDdetector()
        elif control_type == 'canny':
            canny_detector = CannyDetector()
            low_threshold = canny_low
            high_threshold = canny_high

            def apply_canny(x):
                return canny_detector(x, low_threshold, high_threshold)

            self.detector = apply_canny

        elif control_type == 'depth':
            midas = MidasDetector()

            def apply_midas(x):
                detected_map, _ = midas(x)
                return detected_map

            self.detector = apply_midas

global_state = GlobalState()
# model_names = ['revAnimated_v11', 'realisticVisionV20_v20', 'Cyberpunk-Anime-Diffusion', 'Stable_Diffusion_PaperCut_Model']
# assert sd_model in model_names, "choose a unknown model"




global_state.update_sd_model(sd_model, control_type)
global_state.update_controller(0,0,0,0,0)
global_state.update_detector(control_type, low_threshold, high_threshold)


def save_feature_maps(blocks, i, feature_type="input_block"):
    # block_idx = 0
    # for block in blocks:
    #     if feature_type == "input_block" and block_idx in [6,7]:
    #         if "Downsample" in str(type(block[0])) and block_idx:
    #             save_feature_map(block[0].down_output_feature,
    #                              f"down_output_{block_idx}_time_{0}")
    #         if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
    #             save_feature_map(block[1].attention_output,
    #                              f"down_output_{block_idx}_time_{0}")
    #     elif feature_type == "output_block":
    #
    #         if "ResBlock" in str(type(block[0])) and block_idx in [4,5]:
    #             save_feature_map(block[0].out_layers_features,
    #                              f"{feature_type}_{block_idx}_out_layers_features_time_{0}")
    #
    #         if len(block) > 1 and "SpatialTransformer" in str(type(block[1])) and block_idx in [3, 4, 5]:  # block:[resblock, spatial]
    #             save_feature_map(block[1].transformer_blocks[0].attn1.tmp_sim, f"attn_{block_idx}_frame_{0}")
    #     block_idx += 1

    #  save all features
    block_idx = 0
    for block in blocks:
        if feature_type == "input_block":
            if "Downsample" in str(type(block[0])) and block_idx:
                save_feature_map(block[0].down_output_feature,
                                 f"down_output_{block_idx}_time_{0}")
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                save_feature_map(block[1].attention_output,
                                 f"down_output_{block_idx}_time_{0}")
        elif feature_type == "output_block":

            if "ResBlock" in str(type(block[0])):
                save_feature_map(block[0].out_layers_features,
                                 f"{feature_type}_{block_idx}_out_layers_features_time_{0}")

            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):  # block:[resblock, spatial]
                save_feature_map(block[1].transformer_blocks[0].attn1.tmp_sim, f"attn_{block_idx}_frame_{0}")
        block_idx += 1


def save_feature_maps_callback(i, unet_model):
    save_feature_maps(unet_model.input_blocks, i, "input_block")
    save_feature_maps(unet_model.output_blocks, i, "output_block")

if not os.path.exists(feature_path):
    os.makedirs(feature_path, exist_ok=True)

def save_feature_map(feature_map, filename):
    os.makedirs(feature_path, exist_ok=True)
    save_path = os.path.join(feature_path, f"{filename}.pt")
    torch.save(feature_map, save_path)

@torch.no_grad()
def single_inversion(x0, ddim_v_sampler, img_callback=None):
    inv_steps = 1000
    model = ddim_v_sampler.model

    prompt = f""
    cond = {
        'c_concat': None,
        'c_crossattn': [
            model.get_learned_conditioning(
                [prompt]
            )
        ]
    }
    un_cond = {
        'c_concat': None,
        'c_crossattn': [
            model.get_learned_conditioning(
                ['']
            )
        ]
    }
    ddim_v_sampler.encode_ddim(x0, inv_steps, cond, un_cond, img_callback=img_callback)

ddim_v_sampler = global_state.ddim_v_sampler
model = ddim_v_sampler.model
detector = global_state.detector
controller = global_state.controller

model.control_scales = [0.9] * 13
model.to(device)

######### 批量出图  ##########



first_strength = 1 - x0_strength

from tqdm import tqdm

num_samples = 1
unet_model = model.model.diffusion_model
controller.batch_frame_attn_feature_path = feature_path
unet_model.batch_frame_attn_feature_path = feature_path

processed = []
style = ""
if "cartoon" in prompt:
    style = "cartoon"
elif "realistic" in prompt:
    style = "realistic"
else:
    style = "unknown"
# base_dir = "./exp/ImageNet/imagenet-r"
# -------------------------------------------------------------------------------------------
base_dir = "./dataset/coco" 
# base_dir = "./dataset/imagenet-r" 
# base_dir = "./imagenet" 
# -------------------------------------------------------------------------------------------
src_dir_list = os.listdir(base_dir)
# -------------------------------------------------------------------------------------------
# results_dir = f"./exp/ImageNet/batch_results/{style}_SD_ours_{steps}_{control_type}_{base_dir.split('/')[-1]}_{x0_strength}_no_prompt"
# results_dir = f"./exp/ImageNet/batch_results/{sd_model}_{steps}_{control_type}_imagenet-r_{x0_strength}"
# results_dir = f"./exp/ImageNet/batch_results/{sd_model}_{steps}_{control_type}_imagenet_{x0_strength}"
results_dir = f"./exp/ImageNet/batch_results/layer_cmp_{base_dir.split('/')[-1]}"
# -------------------------------------------------------------------------------------------
os.makedirs(results_dir, exist_ok=True)
processed_recorder = os.path.join(results_dir, "recorder.txt")
processed.extend(os.listdir(results_dir))
# if os.path.exists(processed_recorder):
#     with open(processed_recorder, "r") as f:
#         lines = f.readlines()
#         processed = [*lines][:-1]
seed_everything(0)
for i, dir_path in enumerate(tqdm(src_dir_list)):
    # if dir_path in processed:
    #     continue
    # with open(processed_recorder, "a") as f:
    #     f.writelines(dir_path)
    print(f"{i} ", dir_path)
    # src_img_dir = f"./exp/ImageNet/imagenet-r/{dir_path}"
    src_img_dir = os.path.join(base_dir, dir_path)
    results = os.path.join(results_dir, f"{dir_path}")
    src_img_names = os.listdir(src_img_dir)
    src_imgs_path = [os.path.join(src_img_dir, img) for img in src_img_names]
    os.makedirs(results, exist_ok=True)
    with torch.no_grad():
        def generate_first_img(x0, img, strength):
            samples, _ = ddim_v_sampler.sample(
                steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=7.5,
                unconditional_conditioning=un_cond,
                controller=controller,
                x0=x0,
                strength=strength)
            x_samples = model.decode_first_stage(samples)
            x_samples_np = (
                    einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
                    127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            return x_samples, x_samples_np


        for img_name, img_path in zip(src_img_names, src_imgs_path):
            seed_everything(0)
            # print(f"processing {img_name}")
            print(img_name)
            frame = cv2.imread(img_path)
            # frame = cv2.resize(frame,(512,512))
            shape = frame.shape
            if "videogame" in img_name:
                continue
            if shape[0] > 550 or shape[1] > 550 or shape[2] > 550:
                print(f"{img_name} skip")
                continue
            if shape[0] < 300 or shape[1] < 300:
                print(f"{img_name} skip")
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.array(
                transformers.CenterCrop((frame.shape[0] // 64 * 64, frame.shape[1] // 64 * 64))(Image.fromarray(frame)))

            img = HWC3(frame)
            H, W, C = img.shape
            shape = (4, H // 8, W // 8)
            img_ = numpy2tensor(img)

            encoder_posterior = model.encode_first_stage(img_.to(device))
            x0 = model.get_first_stage_encoding(encoder_posterior).detach()
            if use_control:
                detected_map = detector(img)
                detected_map = HWC3(detected_map)
                control = torch.from_numpy(detected_map.copy()).float().to(device) / 255.0
                control = torch.stack([control for _ in range(num_samples)], dim=0)
                control = einops.rearrange(control, 'b h w c -> b c h w').clone()
                cond = {
                    'c_concat': [control],
                    'c_crossattn': [
                        model.get_learned_conditioning(
                            [prompt + ', ' + a_prompt] * num_samples)
                    ]
                }
                un_cond = {
                    'c_concat': [control],
                    'c_crossattn':
                        [model.get_learned_conditioning([n_prompt] * num_samples)]
                }
            else:
                cond = {
                    'c_crossattn': [
                        model.get_learned_conditioning(
                            [prompt + ', ' + a_prompt] * num_samples)
                    ],
                    'c_concat': None
                }
                un_cond = {
                    'c_crossattn':
                        [model.get_learned_conditioning([n_prompt] * num_samples)],
                    'c_concat': None,
                }

            unet_model.unet_type = "denoising"
            controller.set_task('')


            def ddim_sampler_callback(i):
                save_feature_maps_callback(i, unet_model)


            single_inversion(x0, ddim_v_sampler, ddim_sampler_callback)

            Image.fromarray(frame).save(os.path.join(results, img_name))

            seed_everything(0)
            x_samples, x_samples_np = generate_first_img(x0, img, first_strength)
            Image.fromarray(x_samples_np[0]).save(os.path.join(results, img_name[:-4] + "_sd.jpg"))

            controller.set_task('initfirst')
            controller.threshold_block_idx = [3, 4, 5]
            seed_everything(0)
            x_samples, x_samples_np = generate_first_img(x0, img, first_strength)
            Image.fromarray(x_samples_np[0]).save(os.path.join(results, img_name[:-4] + f"_attn{''.join(map(str,controller.threshold_block_idx))}.jpg"))
            controller.set_task('')

            # unet_model.unet_type = "spatial"
            # unet_model.unet_step_th = 0
            # unet_model.unet_block_th = (4, 5)
            # seed_everything(0)
            # x_samples, x_samples_np = generate_first_img(x0, img, first_strength)
            # Image.fromarray(x_samples_np[0]).save(os.path.join(results, img_name[:-4] + "_onlyspatial.jpg"))
            # unet_model.unet_type = "denoising"
            # #
            # unet_model.unet_type = "spatial"
            # unet_model.unet_step_th = 0
            # unet_model.unet_block_th = (4, 5)
            # controller.set_task('initfirst')
            # controller.threshold_block_idx = [3, 4]
            # seed_everything(0)
            # x_samples, x_samples_np = generate_first_img(x0, img, first_strength)
            # Image.fromarray(x_samples_np[0]).save(os.path.join(results, img_name[:-4] + "_all.jpg"))

            controller.set_task('initfirst')
            controller.threshold_block_idx = [5, 6, 7]
            seed_everything(0)
            x_samples, x_samples_np = generate_first_img(x0, img, first_strength)
            Image.fromarray(x_samples_np[0]).save(os.path.join(results, img_name[:-4] + f"_attn{''.join(map(str,controller.threshold_block_idx))}.jpg"))
            controller.set_task('')

            controller.set_task('initfirst')
            controller.threshold_block_idx = [7, 8, 9]
            seed_everything(0)
            x_samples, x_samples_np = generate_first_img(x0, img, first_strength)
            Image.fromarray(x_samples_np[0]).save(
                os.path.join(results, img_name[:-4] + f"_attn{''.join(map(str, controller.threshold_block_idx))}.jpg"))
            controller.set_task('')

            controller.set_task('initfirst')
            controller.threshold_block_idx = [1, 2]
            seed_everything(0)
            x_samples, x_samples_np = generate_first_img(x0, img, first_strength)
            Image.fromarray(x_samples_np[0]).save(
                os.path.join(results, img_name[:-4] + f"_attn{''.join(map(str, controller.threshold_block_idx))}.jpg"))
            controller.set_task('')
