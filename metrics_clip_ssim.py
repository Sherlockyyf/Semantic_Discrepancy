import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer, CLIPProcessor, BlipProcessor, BlipModel, BlipForConditionalGeneration
# from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import cv2
import glob
from skimage import io
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/revAnimated_v11_20/"
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/Stable_Diffusion_PaperCut_Model_20/"
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/realisticVisionV20_v20_30/"
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/Stable_Diffusion_PaperCut_Model_30/"
# trans_img_dir = "/mnt/pami203/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/Stable Diffusion 1.5_20_canny_coco/train2017"
# trans_img_dir = "/mnt/pami203/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/Stable Diffusion 1.5_20_canny_coco_0.5"
# trans_img_dir = "/mnt/pami203/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/Stable Diffusion 1.5_20_canny_coco_0.5_ours/"
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/Stable Diffusion 1.5_20_canny_coco_0.95/"
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/Stable Diffusion 1.5_20_canny_imagenet_0.95/"
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/Stable Diffusion 1.5_20_canny_imagenet_0.95/"
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/Stable Diffusion 1.5_20_canny_imagenet-r_0.95/"
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/Stable Diffusion 1.5_20_canny_imagenet_r_0.6/"
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/Stable Diffusion 1.5_20_canny_imagenet_r_0.7/"
# trans_img_dir = "/mnt/pami203/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/pnp_imagenet-r_sd15/"
# trans_img_dir = "/mnt/pami203/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/pnp_imagenet_sd15/"
# trans_img_dir = "/mnt/pami203/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/pnp_coco_sd15/"
# trans_img_dir = "/mnt/pami203/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/p2p_imagenet_sd15_caption/"
# trans_img_dir = "/mnt/pami203/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/p2p_imagenet-r_sd15_caption/"
# trans_img_dir = "/mnt/pami203/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/p2p_coco_sd15_caption/"
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/Stable Diffusion 1.5_20_canny_imagenet-r_0.2/"
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/Stable Diffusion 1.5_ours_20_canny_coco_0.5/"
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/Stable Diffusion 1.5_ours_20_canny_coco_0.3"
########################################### 20240306 #####################################################################
# pnp_coco
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/pnp_coco_sd15_no_prompt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/pnp_comparison/pnp_coco_style_0306.txt"
# pnp_imagenet
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/pnp_imagenet_sd15_no_prompt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/pnp_comparison/pnp_imagenet_0307.txt"
# # pnp_imagenet-r
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/pnp_imagenet-r_sd15_no_prompt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/pnp_comparison/pnp_imagenet-r_0307.txt"
# p2p_coco
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/cartoon_p2p_coco_sd15_caption_no_prompt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/p2p_comparison/p2p_coco_0307.txt"
# p2p_imagenet
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/cartoon_p2p_imagenet_sd15_caption_no_prompt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/p2p_comparison/p2p_imagenet_0307.txt"
# # p2p_imagenet-r
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/realistic_p2p_imagenet-r_sd15_caption_no_prompt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/p2p_comparison/p2p_imagenet-r_0307.txt"
# ours_imagenet-r_0.7
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/realistic_SD_ours_20_canny_imagenet-r_0.7_no_prompt"
# ours_coco_0.7
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/cartoon_SD_ours_20_canny_coco_0.7_no_prompt" 
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/ours/coco_cartoon_0.7_style_0306.txt"
# ours_coco_0.5
trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/cartoon_SD_ours_20_canny_coco_0.3_no_prompt" 
savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/ours/coco_cartoon_0.3_style_0307.txt"
# ours_imagenet_0.5
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/cartoon_SD_ours_20_canny_imagenet_0.5_no_prompt" 
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/ours/imagenet_cartoon_0.5_style_0307.txt"
# ours_imagnet-r_0.5
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/realistic_SD_ours_20_canny_imagenet-r_0.3_no_prompt" 
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/ours/imagenet-r_realistic_0.3_style_ablation.txt"

# ours_imagnet-r_0.8
# trans_img_dir = "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/realistic_SD_ours_20_canny_imagenet-r_0.5_no_prompt" 
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/ours/imagenet-r_realistic_0.5_style_0307.txt"


# style=None
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/pnp_comparison/pnp_coco_style_0306.txt"

# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/ours/imagenet-r_realistic_0.7_0306.txt"

# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/ours/coco_ours_sd15_0.3.txt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/ours/coco_ours_sd15_0.5.txt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/SDEdit_comparison/coco_sdedit_ours_realistic_0.5.txt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/SDEdit_comparison/imagenet-r_sdedit_ours_0.6.txt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/p2p_comparison/ours_coco_sd15_all.txt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/p2p_comparison/ours_coco_sd15_attn_spa.txt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/p2p_comparison/ours_imagenet_sd15_attn_spa.txt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/p2p_comparison/p2p_imagenet_sd15_2.txt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/pnp_comparison/ours_coco_sd15.txt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/p2p_comparison/p2p_imagenet-r_sd15.txt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/p2p_comparison/p2p_coco_sd15_2.txt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/pnp_comparison/pnp_coco_sd15_3.txt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/pnp_comparison/pnp_imagenet_sd15.txt"
# savefile_path = "/mnt/pami203/yfyuan/YYF/Rerender/exp/METRICS/pnp_comparison/pnp_imagenet-r_sd15_3.txt"
# savefile_path = "/home/yfyuan/YYF/Rerender/exp/METRICS/sd_imagenetr.txt"
# savefile_path = "/home/yfyuan/YYF/Rerender/exp/METRICS/realisticVisionV20_v20_20.txt"
# savefile_path = "/home/yfyuan/YYF/Rerender/exp/METRICS/revAnimated_v11_20.txt"
# savefile_path = "/home/yfyuan/YYF/Rerender/exp/METRICS/papercut_30_2.txt"

print(savefile_path)
ablation = True if "ablation" in savefile_path else None

if "coco" in trans_img_dir:
    dataset = "coco"
    style = "cartoon style of "
elif "imagenet-r" in trans_img_dir:
    dataset = "imagenet-r"
    style = "realistic style of "
elif "imagenet" in trans_img_dir:
    dataset = "imagenet"
    style = "cartoon style of "

if "ours" in trans_img_dir:
    method = "ours"
elif "pnp" in trans_img_dir:
    method = "pnp"
    style = None
elif "p2p" in trans_img_dir:
    method = "p2p"
    style = None
    
print(dataset)
 # Load the CLIP  & BLIP model
clip_model_ID = "/mnt/pami202/blli/pretrained_models/clip-vit-large-patch14"
blip_model_ID = "/mnt/pami202/huggingface/Salesforce/blip-image-captioning-large"
clip_model = CLIPModel.from_pretrained(clip_model_ID, local_files_only=True)
clip_model.cuda()
clip_processor = CLIPProcessor.from_pretrained(clip_model_ID)
print("Load CLIP Success!")
blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_ID, local_files_only=True)
blip_model.cuda()
blip_processor = BlipProcessor.from_pretrained(blip_model_ID)
print("Load BLIP Success!")


def clip_score (gt_path, img1_path, img2_path, style):
    clip_text_2 = 0
    clip_img_2 = 0
    gt = Image.open(gt_path)
    img1 = Image.open(img1_path)
    if img2_path:
        img2 = Image.open(img2_path)
    
    # 生成标题
    gt_inputs = blip_processor(images=gt, return_tensors="pt").to(device, torch.float16)
    gt_outputs = blip_model.generate(**gt_inputs)
    caption = blip_processor.decode(gt_outputs[0], skip_special_tokens=True)
    
    if style:
        caption = style + caption
    
    # Calculate the embeddings for the images using the CLIP model
    if img2_path:
        inputs_images = clip_processor(images=[gt, img1, img2], return_tensors="pt").to(device)
    else:  
        inputs_images = clip_processor(images=[gt, img1], return_tensors="pt").to(device)
   
    with torch.no_grad():
        outputs_images = clip_model.get_image_features(**inputs_images)
        clip_img_1 = torch.cosine_similarity(outputs_images[0].unsqueeze(0), outputs_images[1].unsqueeze(0)).item()
        if img2_path:
            clip_img_2 = torch.cosine_similarity(outputs_images[0].unsqueeze(0), outputs_images[2].unsqueeze(0)).item()
    if img2_path:
        inputs_text = clip_processor(text=[caption], images=[img1, img2], return_tensors="pt", padding=True).to(device)
    else:
        inputs_text = clip_processor(text=[caption], images=[img1], return_tensors="pt", padding=True).to(device)

    outputs_text = clip_model(**inputs_text)
    clip_text_1 = torch.cosine_similarity(outputs_images[1].unsqueeze(0), outputs_text.text_embeds).item()
    if img2_path:
        clip_text_2 = torch.cosine_similarity(outputs_images[2].unsqueeze(0), outputs_text.text_embeds).item()
    
    return clip_img_1, clip_img_2, clip_text_1, clip_text_2, caption

def ssim_score(gt_path, img1_path, img2_path):
    gt_img = io.imread(gt_path)
    H, W = gt_img.shape[0], gt_img.shape[1]
    img1 = io.imread(img1_path)
    img1 = cv2.resize(img1, (W, H))
    ssim_2 = 0
    if img2_path:
        img2 = io.imread(img2_path)
        img2 = cv2.resize(img2, (W, H))

    if len(gt_img.shape) == 3:
        gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_RGB2GRAY)
    else:
        gt_gray = gt_img
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    if img2_path:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
   
    ssim_1 = ssim(gt_gray, img1_gray)
    if img2_path:
        ssim_2 = ssim(gt_gray, img2_gray)

    return ssim_1, ssim_2


# pattern = trans_img_dir + '/*/*_all.jpg'
# sd_files = glob.glob(pattern)
# pattern = trans_img_dir + '/*/*_attn345.jpg'
# pattern = trans_img_dir + '/*_clip/*_pnp.jpg'
if method == "ours":
    if ablation:
        pattern = trans_img_dir + '/*/*_attn345.jpg'
    else:
        pattern = trans_img_dir + '/*/*_sd.jpg'      # ours
elif method == "p2p":
    pattern = trans_img_dir + '/*/*_pnp_clip.jpg'  # p2p
else:
    pattern = trans_img_dir + '/*/*_pnp.jpg'   # pnp
print(method)
print(pattern)
sd_files = glob.glob(pattern, recursive=True)
# print(sd_files)
# sd_files=["/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/Stable_Diffusion_PaperCut_Model_30/saint_bernard/misc_19_sd.jpg", "/home/yfyuan/YYF/Rerender/exp/ImageNet/batch_results/Stable_Diffusion_PaperCut_Model_30/saint_bernard/sketch_0_sd.jpg"]
# print(sd_files)
quan_list = [0,0,0,0,0,0] # gt_sd_img; gt_all_img; gt_sd_text; gt_all_text; ssim_1; ssim_2 

gt_sd_idx = 0
gt_all_idx = 0

for sd_file in sd_files:
    if method == "ours":
        if ablation:
            base_name = sd_file.split('_attn345')[0]
            all_img_path = base_name + "_onlyspatial.jpg"    # ablation
        else:
            base_name = sd_file.split('_sd')[0]   # ours
            all_img_path = base_name + "_all.jpg"   # ours
    elif method == "p2p":
        base_name = sd_file.split('_pnp_clip')[0]   # p2p
        all_img_path = None    # p2p
    else:
        base_name = sd_file.split('_pnp')[0]   # pnp
        all_img_path = None    # pnp/p2p
        
    # base_name = sd_file.split('_attn345')[0]
    # base_name = sd_file.split('_all')[0]
    # base_name = os.path.basename(sd_file).split('_sd')[0]
    
    if dataset == "imagenet":
        base_img_path = base_name + "JPEG"   # imagenet
    else:
        base_img_path = base_name + ".jpg"  # imagenet-r/coco
    
    
    # all_img_path = base_name + "_onlyspatial.jpg"    # ablation
    
    print(f"base_img_path: {base_img_path}")
    print(f"sd_img_path: {sd_file}")
    print(f"all_img_path: {all_img_path}")
    # if os.path.exists(base_img_path) and os.path.exists(all_img_path):
    if os.path.exists(base_img_path):
        gt_sd_img_score, gt_all_img_score, gt_sd_text_score, gt_all_text_score, caption = clip_score(base_img_path, sd_file, all_img_path, style)
        ssim1, ssim2 = ssim_score(base_img_path, sd_file, all_img_path)
        print(f"{base_img_path} and {sd_file}:")
        print(f"Caption: \"{caption}\"")
        # print(f"SD Text: {gt_sd_text_score} All Text: {gt_all_text_score}")
        print(f"SD Img: {gt_sd_img_score} All Img: {gt_all_img_score} SSIM1: {ssim1} SSIM2: {ssim2} SD Text: {gt_sd_text_score} All Text: {gt_all_text_score}")
        quan_list[0] += gt_sd_img_score
        quan_list[1] += gt_all_img_score
        quan_list[2] += gt_sd_text_score
        quan_list[3] += gt_all_text_score
        quan_list[4] += ssim1
        quan_list[5] += ssim2
        # quan_list[0] += gt_sd_text_score
        # quan_list[1] += gt_all_text_score
        gt_sd_idx += 1
        gt_all_idx += 1
        print(gt_all_idx)
        print(quan_list)

print(f"sd num: {gt_sd_idx}")
print(f"all num: {gt_all_idx}")

for i in range(len(quan_list)):
    if i % 2==0:
        quan_list[i] /= gt_sd_idx 
    else:
        quan_list[i] /= gt_all_idx
        
print(quan_list)

with open(savefile_path, "w") as file:
    file.write(f"base file: {base_img_path}\n")
    file.write(f"sd file: {sd_file}\n")
    file.write(f"all file: {all_img_path}\n")
    file.write(f"sd num: {gt_sd_idx}\n")
    file.write(f"all num: {gt_all_idx}\n\n")
    file.write(f"SD Img:  {quan_list[0]}\nAll Img: {quan_list[1]}\nSD Text:  {quan_list[2]}\nAll Text: {quan_list[3]}\nSSIM1: {quan_list[4]}\nSSIM2: {quan_list[5]}")

file.close()
print(savefile_path)
    # file.write("gt_sd_text; gt_all_text\n")
    # for item in quan_list:
    #     file.write("%f\n" % item)
    
