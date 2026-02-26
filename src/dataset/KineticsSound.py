import csv
import os
import pickle
import librosa
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from ..configs import Dataset_Config, CODE_DIR

class KineticsSound(Dataset):
    def __init__(self, args, mode='train', partition = (0, 1)):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        self.data_root = './data/'
        class_dict = {'dribbling_basketball': 0, 'tap_dancing': 1, 'playing_harmonica': 2, 'shoveling_snow': 3, 'singing': 4, 'mowing_lawn': 5, 'tapping_guitar': 6, 'playing_accordion': 7, 'playing_guitar': 8, 'playing_drums': 9, 'playing_trumpet': 10, 'shuffling_cards': 11, 'playing_bass_guitar': 12, 'playing_trombone': 13, 'playing_bagpipes': 14, 'blowing_out_candles': 15, 'playing_organ': 16, 'playing_saxophone': 17, 'bowling': 18, 'blowing_nose': 19, 'playing_piano': 20, 'playing_violin': 21, 'laughing': 22, 'playing_clarinet': 23, 'tapping_pen': 24, 'chopping_wood': 25, 'playing_xylophone': 26, 'playing_keyboard': 27, 'ripping_paper': 28, 'tickling': 29, 'stomping_grapes': 30}
        class_counter = {key:0 for key in class_dict.keys()}
        self.class_dict = class_dict
        
        self.train_csv = os.path.join(self.data_root, args.dataset + '/train.csv')
        self.test_csv = os.path.join(self.data_root, args.dataset + '/test.csv')

        self.visual_feature_path = Dataset_Config().get_dataset_root("Kinetics-Sounds")
        self.audio_feature_path = os.path.join(Dataset_Config().get_dataset_root('Kinetics-Sounds'), "preprocessed_AudioWAV")
            
        if mode == 'train':
            csv_file = self.train_csv
            
        elif mode == 'val':
            csv_file = self.train_csv
        else:
            csv_file = self.test_csv
        
        if os.listdir(self.audio_feature_path)[0].endswith('.pkl'):
            self.audio_type = '.pkl'
        elif os.listdir(self.audio_feature_path)[0].endswith('.wav'):
            self.audio_type = '.wav'
        elif os.listdir(self.audio_feature_path)[0].endswith('.npy'):
            self.audio_type = '.npy'
        else:
            raise ValueError('Audio should be .pkl or .wav or .npy')

        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            next(csv_reader)
            for item in csv_reader:
                if self.audio_type == '.pkl':
                    audio_path = os.path.join(self.audio_feature_path, item[0] + '.pkl')
                elif self.audio_type == '.wav':
                    audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
                elif self.audio_type == '.npy':
                    audio_path = os.path.join(self.audio_feature_path, item[0] + '.npy')
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(args.fps), item[0])

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item[1]])
                    class_counter[item[1]] += 1
                else:
                    continue
        print(f"{len(self.image)} datums in the {mode} dataset") 
        self.start_idx = int(partition[0] * len(self.image))
        self.end_idx = int(partition[1] * len(self.image))
    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        idx += self.start_idx
        # audio
        if self.audio_type == '.pkl':
            with open(self.audio[idx], 'rb') as f2:
                spectrogram = pickle.load(f2)
        elif self.audio_type == '.wav':
            samples, rate = librosa.load(self.audio[idx], sr=22050)
            resamples = np.tile(samples, 10)[:22050*3]
            resamples[resamples > 1.] = 1.
            resamples[resamples < -1.] = -1.

            spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
            spectrogram = np.log(np.abs(spectrogram) + 1e-7)
            #mean = np.mean(spectrogram)
            #std = np.std(spectrogram)
            #spectrogram = np.divide(spectrogram - mean, std + 1e-9)
        elif self.audio_type == '.npy':
            spectrogram = np.load(self.audio[idx])

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        while len(image_samples) < self.args.frames:
            image_samples.append(image_samples[-1])
        images = torch.zeros((self.args.frames, 3, 224, 224))
        for i in range(self.args.frames):
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            images[i] = img
        images = torch.permute(images, (1,0,2,3))

        # label
        label = self.label[idx]

        return spectrogram, images, label, torch.LongTensor([idx])