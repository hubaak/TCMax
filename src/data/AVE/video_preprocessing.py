import pandas as pd
import cv2
import os
import re
import pdb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def get_files(data_dir):
    exist_video = []
    for (root,dirs,files) in os.walk(data_dir, topdown=True):
        for filename in files:
            if filename.endswith('.mp4'):
                exist_video.append((os.path.join(root, filename), filename.replace('.mp4', '')))
    return exist_video

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
            if frame_id<frame_interval*self.frame_kept_per_second and frame_id%frame_interval == 0:
                save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, count)
                cv2.imencode('.jpg', image)[1].tofile(save_name)

            frame_id += 1
            count += 1


class AVE_dataset(object):
    def __init__(self, path_to_dataset = '/data/wuxy/paper/datasets/AVE_Dataset', split='AVE', frame_interval=1, frame_kept_per_second=1):
        self.path_to_video = os.path.join(path_to_dataset, split)
        self.frame_kept_per_second = frame_kept_per_second
        self.path_to_save = os.path.join(path_to_dataset, 'Image-{:02d}-FPS'.format(self.frame_kept_per_second))
        if not os.path.exists(self.path_to_save):
            os.mkdir(self.path_to_save)
        self.file_list = get_files(self.path_to_video)

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
            
        # for i, item in enumerate(tqdm(self.file_list)):
        #     each_video, video_name = item
        #     video_dir = each_video
        #     try:
        #         self.videoReader = videoReader(video_path=video_dir, frame_kept_per_second=self.frame_kept_per_second)

        #         save_dir = os.path.join(self.path_to_save, video_name)
        #         if not os.path.exists(save_dir):
        #             os.mkdir(save_dir)
        #         self.videoReader.video2frame_update(frame_save_path=save_dir)
        #     except:
        #         print('Fail @ {}'.format(each_video[:-1]))


ave = AVE_dataset()
ave.extractImage()

