import os
import librosa
import numpy as np
import pickle
import csv
from tqdm import tqdm
import random
from ...configs import Dataset_Config
from concurrent.futures import ThreadPoolExecutor, as_completed

dataset_path = Dataset_Config().get_dataset_root("CREMA-D")

dataset_config = {
    'audio_path' : 'AudioWAV',
    'out_path' : 'preprocessed_AudioWAV',
}

IF_REMOVE_ORIGIN = False

def process_audio(path):
    rate = 22050
    samples, rate = librosa.load(path, sr=22050)
    if len(samples)==0:
        return None
    if len(samples) < rate*10:
        resamples = np.tile(samples, 1 + rate*10 // len(samples))
    else:
        resamples = samples
    start_point = random.randint(a=0, b=rate*5)
    resamples = resamples[start_point : start_point + rate * 5]
    resamples[resamples > 1.] = 1.
    resamples[resamples < -1.] = -1.

    spectrogram = librosa.stft(resamples, n_fft=256, hop_length=128)
    spectrogram = np.log(np.abs(spectrogram) + 1e-7)
    return spectrogram

def get_path_from_csv(path_to_csv, dataset_p, dataset_out):
    data_path = []
    with open(path_to_csv) as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for item in csv_reader:
            audio_path = os.path.join(dataset_p, item[0] + '.wav')
            output_path = os.path.join(dataset_out, item[0] + '.pkl')
            # assert os.path.exists(audio_path)
            if os.path.exists(audio_path):
                data_path.append((audio_path, output_path))
    return data_path

def get_path_from_dir(dataset_p, dataset_out):
    data_path = []
    all_files = os.listdir(dataset_p)
    for item in all_files:
        audio_path = os.path.join(dataset_p, item)
        output_path = os.path.join(dataset_out, item.replace('.wav', '.pkl'))
        # assert os.path.exists(audio_path)
        if os.path.exists(audio_path):
            data_path.append((audio_path, output_path))
    return data_path

# def preprocessed_and_save(data_path):
#     for in_path, out_path in tqdm(data_path):
#         if os.path.exists(out_path):
#             continue
#         spectrogram = process_audio(in_path)
#         if spectrogram is not None:
#             with open(out_path, 'wb') as f:
#                 pickle.dump(spectrogram, f)
#         if IF_REMOVE_ORIGIN:
#             os.remove(in_path)

def process_single_item(in_path, out_path):
    if os.path.exists(out_path):
        return 
    try:
        spectrogram = process_audio(in_path)
        if spectrogram is not None:
            with open(out_path, 'wb') as f:
                pickle.dump(spectrogram, f)
        if IF_REMOVE_ORIGIN:
            os.remove(in_path)
    except Exception as e:
        print(f"处理文件 {in_path} 出错: {e}")

def preprocessed_and_save(data_path, max_workers=None):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_item, in_path, out_path) 
                  for in_path, out_path in data_path]
        
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing audio"):
            pass
        
        
if __name__ == "__main__":
    audio_dataset_p = os.path.join(dataset_path, dataset_config['audio_path'])
    audio_out_dataset_p = os.path.join(dataset_path, dataset_config['out_path'])
    os.makedirs(audio_out_dataset_p, exist_ok=True)
    audio_path = get_path_from_dir(audio_dataset_p, audio_out_dataset_p)
    print(len(audio_path))
    preprocessed_and_save(audio_path, max_workers=24)