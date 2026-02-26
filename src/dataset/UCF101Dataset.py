import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from ..configs import Dataset_Config, CODE_DIR

default_data_folder = {
    'RGB' : 'RGB',
    'OF' : 'OF'
}

class UCF101Dataset(Dataset):
    def __init__(self, args, mode='train', partition = (0, 1)):
        self.args = args
        self.image = []
        self.Optic_Flow = []
        self.label = []
        self.mode = mode

        self.data_root = os.path.join(CODE_DIR, 'data')
        
        class_dict = {'SalsaSpin': 0, 'Kayaking': 1, 'FrisbeeCatch': 2, 'BoxingSpeedBag': 3, 'FrontCrawl': 4, 'PushUps': 5, 'PoleVault': 6, 'Billiards': 7, 'SumoWrestling': 8, 'CleanAndJerk': 9, 'BenchPress': 10, 'PlayingTabla': 11, 'ApplyEyeMakeup': 12, 'SkyDiving': 13, 'WallPushups': 14, 'BreastStroke': 15, 'Lunges': 16, 'ParallelBars': 17, 'HorseRiding': 18, 'TaiChi': 19, 'SkateBoarding': 20, 'CricketBowling': 21, 'Swing': 22, 'PlayingPiano': 23, 'Drumming': 24, 'FloorGymnastics': 25, 'CliffDiving': 26, 'SoccerPenalty': 27, 'PlayingFlute': 28, 'MilitaryParade': 29, 'IceDancing': 30, 'Knitting': 31, 'CuttingInKitchen': 32, 'HorseRace': 33, 'Shotput': 34, 'HandstandWalking': 35, 'PizzaTossing': 36, 'ThrowDiscus': 37, 'Basketball': 38, 'HammerThrow': 39, 'PlayingDaf': 40, 'BandMarching': 41, 'PlayingCello': 42, 'HighJump': 43, 'RockClimbingIndoor': 44, 'Nunchucks': 45, 'Hammering': 46, 'Rowing': 47, 'HulaHoop': 48, 'TableTennisShot': 49, 'HeadMassage': 50, 'TrampolineJumping': 51, 'BodyWeightSquats': 52, 'Biking': 53, 'BoxingPunchingBag': 54, 'WalkingWithDog': 55, 'BabyCrawling': 56, 'ShavingBeard': 57, 'Bowling': 58, 'PommelHorse': 59, 'BalanceBeam': 60, 'RopeClimbing': 61, 'BaseballPitch': 62, 'PullUps': 63, 'CricketShot': 64, 'UnevenBars': 65, 'PlayingGuitar': 66, 'JugglingBalls': 67, 'PlayingDhol': 68, 'Punch': 69, 'BlowingCandles': 70, 'YoYo': 71, 'GolfSwing': 72, 'LongJump': 73, 'Mixing': 74, 'JumpingJack': 75, 'Skijet': 76, 'HandStandPushups': 77, 'BrushingTeeth': 78, 'Fencing': 79, 'JavelinThrow': 80, 'WritingOnBoard': 81, 'Diving': 82, 'ApplyLipstick': 83, 'SoccerJuggling': 84, 'StillRings': 85, 'Skiing': 86, 'TennisSwing': 87, 'BasketballDunk': 88, 'Rafting': 89, 'Typing': 90, 'MoppingFloor': 91, 'PlayingSitar': 92, 'Surfing': 93, 'Archery': 94, 'PlayingViolin': 95, 'BlowDryHair': 96, 'JumpRope': 97, 'Haircut': 98, 'VolleyballSpiking': 99, 'FieldHockeyPenalty': 100}
        class_counter = {key:0 for key in class_dict.keys()}
        self.class_dict = class_dict
        
        self.train_csv = os.path.join(self.data_root, 'UCF101/train.csv')
        self.test_csv = os.path.join(self.data_root, 'UCF101/test.csv')
        
        dataset_path = Dataset_Config().get_dataset_root("UCF101")
        
        if mode == 'train':
            csv_file = self.train_csv
            self.RGB_feature_path = os.path.join(dataset_path, default_data_folder['RGB'])
            self.OF_feature_path = os.path.join(dataset_path, default_data_folder['OF'])
        elif mode == 'val':
            csv_file = self.train_csv
            self.RGB_feature_path = os.path.join(dataset_path, default_data_folder['RGB'])
            self.OF_feature_path = os.path.join(dataset_path, default_data_folder['OF'])
        else:
            csv_file = self.test_csv
            self.RGB_feature_path = os.path.join(dataset_path, default_data_folder['RGB'])
            self.OF_feature_path = os.path.join(dataset_path, default_data_folder['OF'])
        
        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            # next(csv_reader)
            for item in csv_reader:
                visual_path = os.path.join(self.RGB_feature_path, 'Image-{:02d}-FPS'.format(args.fps), item[0])
                OF_path = os.path.join(self.OF_feature_path, 'Image-{:02d}-FPS'.format(args.fps), item[0])
                
                if os.path.exists(visual_path) and os.path.exists(OF_path):
                    self.image.append(visual_path)
                    self.Optic_Flow.append(OF_path)
                    self.label.append(class_dict[item[1]])
                    class_counter[item[1]] += 1
                else:
                    continue
        self.start_idx = int(partition[0] * len(self.image))
        self.end_idx = int(partition[1] * len(self.image))
        print(f"{self.end_idx - self.start_idx} datums in the {mode} dataset") 
        
        # Visual transform
        if self.mode == 'train':
            if 'CLIP' in args.visual_backbone:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
                ])
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            if 'CLIP' in args.visual_backbone:
                self.transform = transforms.Compose([
                    transforms.Resize(224),  # 调整短边为 224，保持纵横比
                    transforms.CenterCrop(224),  # 中心裁剪成 224x224
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),  # CLIP 的归一化参数
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        self.of_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(224, 224)),
        ])
        
    def __len__(self):
        return self.end_idx - self.start_idx

    def __getitem__(self, idx):
        idx += self.start_idx

        # Visual
        image_samples = os.listdir(self.image[idx])
        while len(image_samples) < self.args.frames:
            image_samples.append(image_samples[-1])
        images = torch.zeros((self.args.frames, 3, 224, 224))
        for i in range(self.args.frames):
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            img = self.transform(img)
            images[i] = img
        images = torch.permute(images, (1,0,2,3))
        
        # OF
        self.OF_type = '.jpg' if os.listdir(self.Optic_Flow[idx])[0].endswith('.jpg') else '.npy'
        if self.OF_type == '.jpg':
            image_samples = os.listdir(self.Optic_Flow[idx])
            while len(image_samples) < self.args.frames:
                image_samples.append(image_samples[-1])
            Optic_Flow = torch.zeros((self.args.frames, 3, 224, 224))
            for i in range(self.args.frames):
                img = Image.open(os.path.join(self.Optic_Flow[idx], image_samples[i])).convert('RGB')
                img = self.transform(img)
                Optic_Flow[i] = img
            Optic_Flow = torch.permute(Optic_Flow, (1,0,2,3))
        else:
            image_samples = os.listdir(self.Optic_Flow[idx])
            while len(image_samples) < self.args.OF_frames:
                image_samples.append(image_samples[-1])
            Optic_Flow = torch.zeros((self.args.OF_frames, 2, 224, 224))
            for i in range(self.args.OF_frames):
                img = np.load(os.path.join(self.Optic_Flow[idx], image_samples[i]))
                img = torch.from_numpy(img).float()
                img = torch.permute(img, (2, 1, 0)).unsqueeze(0)
                img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
                Optic_Flow[i] = img.squeeze(0)
            Optic_Flow = torch.permute(Optic_Flow, (1,0,2,3))
        Optic_Flow = Optic_Flow.reshape(2*self.args.OF_frames, 1, 224, 224)
        # label
        label = self.label[idx]

        return {
            'images1' : images,
            'images2': Optic_Flow, 
            'label' : label,
            'idx' : torch.LongTensor([idx])
        }