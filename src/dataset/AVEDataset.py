import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from ..configs import Dataset_Config, CODE_DIR

class AVEDataset(Dataset):
    def __init__(self, args, mode='train', partition = (0, 1)):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        self.data_root = './data/'
        self.dataset_root = Dataset_Config().get_dataset_root('AVE')
        class_dict = {'Flute': 0, 'Mandolin': 1, 'Bark': 2, 'Helicopter': 3, 'Ukulele': 4, 'Shofar': 5, 'Truck': 6, 'Goat': 7, 'Acoustic guitar': 8, 'Toilet flush': 9, 'Banjo': 10, 'Bus': 11, 'Motorcycle': 12, 'Rodents, rats, mice': 13, 'Female speech, woman speaking': 14, 'Fixed-wing aircraft, airplane': 15, 'Chainsaw': 16, 'Frying (food)': 17, 'Violin, fiddle': 18, 'Cat': 19, 'Race car, auto racing': 20, 'Male speech, man speaking': 21, 'Baby cry, infant cry': 22, 'Clock': 23, 'Church bell': 24, 'Accordion': 25, 'Train horn': 26, 'Horse': 27}
        class_counter = {key:0 for key in class_dict.keys()}
        self.class_dict = class_dict
        
        self.train_csv = os.path.join(self.data_root, args.dataset + '/train.csv')
        self.val_csv = os.path.join(self.data_root, args.dataset + '/val.csv')
        self.test_csv = os.path.join(self.data_root, args.dataset + '/test.csv')

        if mode == 'train':
            csv_file = self.train_csv
            # self.visual_feature_path = args.visual_path
            # self.audio_feature_path = args.audio_path
            self.visual_feature_path = self.dataset_root
            self.audio_feature_path = os.path.join(self.dataset_root, 'preprocessed_AudioWAV')
        elif mode == 'val':
            csv_file = self.val_csv
            self.visual_feature_path = self.dataset_root
            self.audio_feature_path = os.path.join(self.dataset_root, 'preprocessed_AudioWAV')
        else:
            csv_file = self.test_csv
            # self.visual_feature_path = args.test_visual_path
            # self.audio_feature_path = args.test_audio_path
            self.visual_feature_path = self.dataset_root
            self.audio_feature_path = os.path.join(self.dataset_root, 'preprocessed_AudioWAV')
        
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
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(1), item[0])

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