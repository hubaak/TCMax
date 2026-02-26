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


class CREMADataset(Dataset):

    def __init__(self, args, mode='train', partition = (0, 1)):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.bucket = []
        self.mode = mode

        self.data_root = os.path.join(CODE_DIR, 'data')
        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}
        self.class_dict = class_dict

        self.visual_feature_path = Dataset_Config().get_dataset_root('CREMA-D')
        self.audio_feature_path = os.path.join(Dataset_Config().get_dataset_root('CREMA-D'), "preprocessed_AudioWAV")

        self.train_csv = os.path.join(self.data_root, 'CREMAD/train.csv')
        self.test_csv = os.path.join(self.data_root, 'CREMAD/test.csv')

        if mode == 'train':
            csv_file = self.train_csv
        elif mode == 'val':
            csv_file = self.train_csv
        else:
            csv_file = self.test_csv

        if os.listdir(self.audio_feature_path)[0].endswith('.pkl'):
            self.if_audio_pkl = True
        elif os.listdir(self.audio_feature_path)[0].endswith('.wav'):
            self.if_audio_pkl = False

        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                if self.if_audio_pkl:
                    audio_path = os.path.join(self.audio_feature_path, item[0] + '.pkl')
                else:
                    audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(args.fps), item[0])

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item[1]])
                    self.bucket.append(0)
                else:
                    continue
        
        
        self.start_idx = int(partition[0] * len(self.image))
        self.end_idx = int(partition[1] * len(self.image))
        print(f"{self.end_idx - self.start_idx} datums in the {mode} dataset") 
            
    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        idx += self.start_idx
        # audio
        if self.if_audio_pkl:
            with open(self.audio[idx], 'rb') as f2:
                spectrogram = pickle.load(f2)
        else:
            samples, rate = librosa.load(self.audio[idx], sr=22050)
            resamples = np.tile(samples, 10)[:22050*3]
            resamples[resamples > 1.] = 1.
            resamples[resamples < -1.] = -1.

            spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
            spectrogram = np.log(np.abs(spectrogram) + 1e-7)
            #mean = np.mean(spectrogram)
            #std = np.std(spectrogram)
            #spectrogram = np.divide(spectrogram - mean, std + 1e-9)

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