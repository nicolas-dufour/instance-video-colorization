import numpy as np
import cv2

def compute_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def compute_video_psnr(video1_path,video2_path):
    vidcap = cv2.VideoCapture(video1_path)
    success,image = vidcap.read()
    frame_list1 = [image]
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        frame_list1.append(image)
        count += 1
    
    vidcap = cv2.VideoCapture(video2_path)
    success,image = vidcap.read()
    frame_list2 = [image]
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        frame_list2.append(image)
        count += 1
    psnr = 0
    for (frames1, frames2) in list(zip(frame_list1,frame_list2):
        psnr+=compute_psnr(frames1, frames2)
    psnr = psnr/len(frames1)
    return psnr