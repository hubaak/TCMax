import os
import re
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from ...configs import Dataset_Config

NUM_THREAD = 64

def process_video(item, output_dir):
    video_filename = item[0]
    wav_filename = os.path.join(output_dir, item[1] + '.wav')
    
    if os.path.exists(wav_filename):
        return f"{wav_filename} exist, skip"
    else:
        subprocess.run(
            ['ffmpeg', '-i', video_filename, '-acodec', 'pcm_s16le', '-ar', '16000', wav_filename],
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        return f"{wav_filename} complete"

def convert_videos_multithread(exist_video, output_dir, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(exist_video)) as pbar:
            futures = [executor.submit(process_video, item, output_dir) for item in exist_video]
            
            for future in futures:
                future.add_done_callback(lambda p: pbar.update(1))
                

def extract_wav(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Search for videos
    exist_video = []
    for (root,dirs,files) in os.walk(data_dir, topdown=True):
        pattern = r'(\d+)_(\d+)_out.mkv'
        for filename in files:
            if filename.find('.mkv') != -1 and filename[0] == 'v':
                match = re.match(pattern, filename[13:])
                if match:
                    start_time = match.group(1)
                    exist_video.append((os.path.join(root, filename), filename.replace('.mkv', '')))
    print('{} videos found in {}, start processing'.format(len(exist_video), data_dir))
    convert_videos_multithread(exist_video, output_dir, max_workers=NUM_THREAD)
            

videos_path = Dataset_Config().get_dataset_root("VGGSound")
data_split = ['train', 'test']

for split in data_split:
    extract_wav(os.path.join(videos_path, split), os.path.join(videos_path, "audio_{}".format(split)))




