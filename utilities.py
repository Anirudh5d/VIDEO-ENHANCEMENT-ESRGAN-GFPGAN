import os
import cv2
import torch
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from basicsr.utils import imwrite
import requests
def download_file(url, destination, expected_size=None):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded: {destination}")
        

        if expected_size and os.path.getsize(destination) != expected_size:
            print(f"File size mismatch for {destination}. Expected {expected_size}, got {os.path.getsize(destination)}. Re-downloading...")
            os.remove(destination)  
            download_file(url, destination, expected_size)
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")


ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()
    frames = []
    while success:
        frames.append(frame)
        success, frame = vidcap.read()
    vidcap.release()
    return frames

def save_video(frames, output_path, fps=30):
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()

def load_models(version='1.3', bg_upsampler_name='realesrgan', upscale=2, bg_tile=400):
    model_dir = 'experiments/pretrained_models'
    os.makedirs(model_dir, exist_ok=True)
    if version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    else:
        raise ValueError(f"Unsupported version: {version}")


    if bg_upsampler_name == 'realesrgan':
        bg_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=upscale,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=bg_model,
            tile=bg_tile,
            tile_pad=10,
            pre_pad=0,
            half=False)  
    else:
        bg_upsampler = None
    arch = 'clean'
    channel_multiplier = 2
    model_name = f'GFPGANv{version}'
    model_path = os.path.join(model_dir, f'{model_name}.pth')
     
    expected_size = 348632874 # this is just to ensure that the model is completely downloaded

    # Check if GFPGAN model is already downloaded and valid
    if not os.path.exists(model_path) or os.path.getsize(model_path) != expected_size:
        url = f'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/{model_name}.pth'
        download_file(url, model_path, expected_size)

    restorer = GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)
    
    return restorer, bg_upsampler

def enhance_video(video_path, output_path, version='1.3', upscale=2, bg_upsampler_name='realesrgan', fps=30, weight=0.5, only_center_face=False, aligned=False):
    restorer, bg_upsampler = load_models(version, bg_upsampler_name, upscale)

    frames = extract_frames(video_path)
    enhanced_frames = [restorer.enhance(frame, aligned, only_center_face, True, weight)[2] for frame in frames]
    
    save_video(enhanced_frames, output_path, fps)
