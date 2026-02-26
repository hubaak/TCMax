import os
import librosa
import numpy as np
import pickle
import csv
from tqdm import tqdm
import random
import torch
import torchaudio
from os.path import join as join
import numpy as np
from ...configs import Dataset_Config

dataset_path = Dataset_Config().get_dataset_root("Kinetics-Sounds")
dataset_config = {
    'train_meta' : 'train.csv',
    'test_meta' : 'test.csv',
    'train_audio_path' : 'AudioWAV',
    'test_audio_path' : 'AudioWAV',
    'out_train_path' : 'preprocessed_AudioWAV',
    'out_test_path' : 'preprocessed_AudioWAV'
}


def wav2fbank(filename, filename2=None, mix_lambda=-1):
        # no mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, 
                                                      use_energy=False, window_type='hanning', 
                                                      num_mel_bins=128, dither=0.0, frame_shift=10)
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            print('there is a loading error')

        target_length = 1024
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

def process_audio(path):
    rate = 16000
    samples, rate = librosa.load(path, sr=rate, mono=True)
    
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

def preprocessed_and_save(data_path):
    for in_path, out_path in tqdm(data_path):
        fbank = wav2fbank(in_path)
        np.save(out_path.replace('.pkl',''), fbank, allow_pickle = True)

if __name__ == "__main__":
    train_meta_path = os.path.join(dataset_path, dataset_config['train_meta'])
    train_audio_dataset_p = os.path.join(dataset_path, dataset_config['train_audio_path'])
    train_audio_out_dataset_p = os.path.join(dataset_path, dataset_config['out_train_path'])
    os.makedirs(train_audio_out_dataset_p, exist_ok=True)
    train_audio_path = get_path_from_dir(train_audio_dataset_p, train_audio_out_dataset_p)
    print(len(train_audio_path))
    preprocessed_and_save(train_audio_path)
    
    test_meta_path = os.path.join(dataset_path, dataset_config['test_meta'])
    test_audio_dataset_p = os.path.join(dataset_path, dataset_config['test_audio_path'])
    test_audio_out_dataset_p = os.path.join(dataset_path, dataset_config['out_test_path'])
    os.makedirs(test_audio_out_dataset_p, exist_ok=True)
    test_audio_path = get_path_from_dir(test_audio_dataset_p, test_audio_out_dataset_p)
    print(len(test_audio_path))
    preprocessed_and_save(test_audio_path)