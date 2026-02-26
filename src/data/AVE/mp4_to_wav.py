import os
import re
from tqdm import tqdm


def extract_wav(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Search for videos
    exist_video = []
    for (root,dirs,files) in os.walk(data_dir, topdown=True):
        for filename in files:
            if filename.endswith('.mp4'):
                exist_video.append((os.path.join(root, filename), filename.replace('.mp4', '')))
    print('{} videos found in {}, start processing'.format(len(exist_video), data_dir))
    for i, item in enumerate(exist_video):
        video_filename = item[0]
        wav_filename = os.path.join(output_dir, item[1]+'.wav')
        if os.path.exists(wav_filename):
            pass
        else:
            os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(video_filename, wav_filename))
            

videos_path = '/mnt/data_2/wuxy/datasets/AVE_Dataset'
data_split = ['AVE']

for split in data_split:
    extract_wav(os.path.join(videos_path, split), os.path.join(videos_path, "audio_{}".format(split)))




