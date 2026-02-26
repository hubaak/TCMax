import pandas as pd
import cv2
import os
import pdb
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ...configs import Dataset_Config

def draw_optical_flow(flow):
    # Calculate magnitude and angle of flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Normalize magnitude to the range [0, 1]
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = np.uint8(magnitude)

    # Create an HSV image
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = magnitude  # Value

    # Convert HSV to RGB
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb_flow

class videoReader(object):
    def __init__(self, video_path, frame_interval=1, frame_kept_per_second=1):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.frame_kept_per_second = frame_kept_per_second

        #pdb.set_trace()
        self.vid = cv2.VideoCapture(self.video_path)
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
        self.video_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_len = int(self.video_frames/self.fps)


    def video2frame(self, frame_save_path):
        self.frame_save_path = frame_save_path
        success, image = self.vid.read()
        count = 0
        while success:
            count +=1
            if count % self.frame_interval == 0:
                save_name = '{}/frame_{}_{}.jpg'.format(self.frame_save_path, int(count/self.fps), count)  # filename_second_index
                cv2.imencode('.jpg', image)[1].tofile(save_name)
            success, image = self.vid.read()

    def video2frame_update(self, frame_save_path):
        self.frame_save_path = frame_save_path
        count = 0
        frame_interval = int(self.fps/self.frame_kept_per_second)
        while(count < self.video_frames):
            ret, image = self.vid.read()
            if not ret:
                break
    
            if count % self.fps == 0:
                    frame_id = 0
                    
            if count != 0:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if frame_id<frame_interval*self.frame_kept_per_second and frame_id%frame_interval == 0:
                    save_name = '{0}/{1:05d}.npy'.format(self.frame_save_path, count)
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    # flow = flow * self.fps / 255.0
                    flow = flow.astype(np.float16)
                    # print(np.max(np.abs(flow)))
                    np.save(save_name, flow)
                    
                    # flow_img = draw_optical_flow(flow)
                    # save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, count)
                    # cv2.imencode('.jpg', flow_img)[1].tofile(save_name)
                prev_gray = gray
            else:
                prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
            frame_id += 1
            count += 1


class UCF101_dataset(object):
    def __init__(self, path_to_dataset = Dataset_Config().get_dataset_root("UCF101"), frame_interval=1, frame_kept_per_second=1):
        self.path_to_video = os.path.join(path_to_dataset, 'UCF-101')
        self.path_to_images = os.path.join(path_to_dataset, 'OF')
        self.frame_kept_per_second = frame_kept_per_second
        self.path_to_save = os.path.join(self.path_to_images, 'Image-{:02d}-FPS'.format(self.frame_kept_per_second))
        if not os.path.exists(self.path_to_images):
            os.mkdir(self.path_to_images)
        if not os.path.exists(self.path_to_save):
            os.mkdir(self.path_to_save)
        self.file_list = self.get_all_files(self.path_to_video)

    def get_all_files(self, path):
        all_files = []
        for (root,dirs,files) in os.walk(path, topdown=True):
            for filename in files:
                if filename.endswith('.avi'):
                    all_files.append((os.path.join(root, filename), filename.replace('.avi', '')))
        return all_files
    
    def extractImage(self):
        def process_video(each_video):
            video_dir = os.path.join(self.path_to_video, each_video[0])
            _videoReader = videoReader(video_path=video_dir, frame_kept_per_second=self.frame_kept_per_second)

            save_dir = os.path.join(self.path_to_save, each_video[1])
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            _videoReader.video2frame_update(frame_save_path=save_dir)
        with ThreadPoolExecutor(max_workers=32) as executor:
            list(tqdm(executor.map(process_video, self.file_list), total=len(self.file_list)))   
            # print('Precessing {} ...'.format(each_video))
            


UCF101 = UCF101_dataset(frame_kept_per_second = 1)
UCF101.extractImage()