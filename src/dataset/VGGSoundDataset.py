import copy
import csv
import os
import pickle
import librosa
import numpy as np
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from ..configs import Dataset_Config, CODE_DIR


class VGGSound(Dataset):

    def __init__(self, args, mode='train', partition = (0, 1)):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode

        self.data_root = './data/'
        
        class_dict = {'chopping food': 0, 'firing muskets': 1, 'pigeon, dove cooing': 2, 'playing electric guitar': 3, 'rapping': 4, 'stream burbling': 5, 'volcano explosion': 6, 'playing tabla': 7, 'tapping guitar': 8, 'baby babbling': 9, 'goat bleating': 10, 'cattle, bovinae cowbell': 11, 'dog whimpering': 12, 'playing acoustic guitar': 13, 'people coughing': 14, 'playing erhu': 15, 'playing castanets': 16, 'playing djembe': 17, 'penguins braying': 18, 'squishing water': 19, 'playing washboard': 20, 'tap dancing': 21, 'people eating crisps': 22, 'mosquito buzzing': 23, 'playing tympani': 24, 'railroad car, train wagon': 25, 'thunder': 26, 'playing bongo': 27, 'chopping wood': 28, 'people eating noodle': 29, 'engine accelerating, revving, vroom': 30, 'dinosaurs bellowing': 31, 'mouse clicking': 32, 'church bell ringing': 33, 'lions growling': 34, 'running electric fan': 35, 'chimpanzee pant-hooting': 36, 'missile launch': 37, 'people eating apple': 38, 'ocean burbling': 39, 'turkey gobbling': 40, 'playing synthesizer': 41, 'train horning': 42, 'sharpen knife': 43, 'typing on computer keyboard': 44, 'mynah bird singing': 45, 'playing tuning fork': 46, 'reversing beeps': 47, 'driving snowmobile': 48, 'dog howling': 49, 'playing sitar': 50, 'typing on typewriter': 51, 'snake rattling': 52, 'child singing': 53, 'bird squawking': 54, 'basketball bounce': 55, 'car engine knocking': 56, 'firing cannon': 57, 'driving buses': 58, 'striking pool': 59, 'goose honking': 60, 'people marching': 61, 'chicken crowing': 62, 'playing congas': 63, 'people cheering': 64, 'cat caterwauling': 65, 'helicopter': 66, 'elephant trumpeting': 67, 'mouse squeaking': 68, 'playing harpsichord': 69, 'playing vibraphone': 70, 'sheep bleating': 71, 'opening or closing drawers': 72, 'dog growling': 73, 'sea lion barking': 74, 'pig oinking': 75, 'bathroom ventilation fan running': 76, 'lawn mowing': 77, 'eletric blender running': 78, 'zebra braying': 79, 'wood thrush calling': 80, 'warbler chirping': 81, 'baltimore oriole calling': 82, 'car engine starting': 83, 'dog bow-wow': 84, 'donkey, ass braying': 85, 'bird chirping, tweeting': 86, 'lip smacking': 87, 'electric shaver, electric razor shaving': 88, 'lighting firecrackers': 89, 'elk bugling': 90, 'wind noise': 91, 'chinchilla barking': 92, 'playing snare drum': 93, 'playing flute': 94, 'people hiccup': 95, 'fox barking': 96, 'people farting': 97, 'playing gong': 98, 'vacuum cleaner cleaning floors': 99, 'strike lighter': 100, 'race car, auto racing': 101, 'cuckoo bird calling': 102, 'footsteps on snow': 103, 'air conditioning noise': 104, 'cow lowing': 105, 'playing didgeridoo': 106, 'skiing': 107, 'foghorn': 108, 'cheetah chirrup': 109, 'people giggling': 110, 'roller coaster running': 111, 'pheasant crowing': 112, 'playing bugle': 113, 'cricket chirping': 114, 'chainsawing trees': 115, 'alligators, crocodiles hissing': 116, 'cupboard opening or closing': 117, 'police radio chatter': 118, 'driving motorcycle': 119, 'people whispering': 120, 'playing violin, fiddle': 121, 'airplane flyby': 122, 'playing bagpipes': 123, 'bouncing on trampoline': 124, 'playing darts': 125, 'horse clip-clop': 126, 'playing mandolin': 127, 'opening or closing car electric windows': 128, 'train whistling': 129, 'fireworks banging': 130, 'fire truck siren': 131, 'playing drum kit': 132, 'eating with cutlery': 133, 'waterfall burbling': 134, 'splashing water': 135, 'planing timber': 136, 'tractor digging': 137, 'people finger snapping': 138, 'rope skipping': 139, 'singing bowl': 140, 'female speech, woman speaking': 141, 'cat growling': 142, 'bowling impact': 143, 'slot machine': 144, 'playing piano': 145, 'eagle screaming': 146, 'playing trumpet': 147, 'child speech, kid speaking': 148, 'bee, wasp, etc. buzzing': 149, 'sliding door': 150, 'popping popcorn': 151, 'playing lacrosse': 152, 'forging swords': 153, 'civil defense siren': 154, 'people screaming': 155, 'car passing by': 156, 'playing harp': 157, 'francolin calling': 158, 'orchestra': 159, 'hail': 160, 'people babbling': 161, 'playing cymbal': 162, 'playing shofar': 163, 'people burping': 164, 'owl hooting': 165, 'otter growling': 166, 'pumping water': 167, 'swimming': 168, 'cat hissing': 169, 'ripping paper': 170, 'people belly laughing': 171, 'train wheels squealing': 172, 'people sneezing': 173, 'police car (siren)': 174, 'duck quacking': 175, 'dog barking': 176, 'telephone bell ringing': 177, 'people clapping': 178, 'using sewing machines': 179, 'singing choir': 180, 'fire crackling': 181, 'shot football': 182, 'black capped chickadee calling': 183, 'smoke detector beeping': 184, 'rowboat, canoe, kayak rowing': 185, 'ambulance siren': 186, 'cell phone buzzing': 187, 'barn swallow calling': 188, 'people running': 189, 'playing tambourine': 190, 'air horn': 191, 'playing cornet': 192, 'yodelling': 193, 'playing marimba, xylophone': 194, 'lions roaring': 195, 'playing table tennis': 196, 'playing badminton': 197, 'spraying water': 198, 'machine gun shooting': 199, 'skidding': 200, 'people sniggering': 201, 'sloshing water': 202, 'cat purring': 203, 'printer printing': 204, 'fly, housefly buzzing': 205, 'plastic bottle crushing': 206, 'playing tennis': 207, 'playing banjo': 208, 'playing accordion': 209, 'playing steel guitar, slide guitar': 210, 'playing oboe': 211, 'people nose blowing': 212, 'gibbon howling': 213, 'canary calling': 214, 'people eating': 215, 'playing theremin': 216, 'raining': 217, 'frog croaking': 218, 'playing volleyball': 219, 'playing french horn': 220, 'people whistling': 221, 'metronome': 222, 'magpie calling': 223, 'male speech, man speaking': 224, 'disc scratching': 225, 'playing harmonica': 226, 'playing hockey': 227, 'playing hammond organ': 228, 'scuba diving': 229, 'male singing': 230, 'striking bowling': 231, 'parrot talking': 232, 'blowtorch igniting': 233, 'people booing': 234, 'people crowd': 235, 'people shuffling': 236, 'playing double bass': 237, 'skateboarding': 238, 'cap gun shooting': 239, 'golf driving': 240, 'people battle cry': 241, 'horse neighing': 242, 'playing zither': 243, 'arc welding': 244, 'children shouting': 245, 'toilet flushing': 246, 'hedge trimmer running': 247, 'airplane': 248, 'tornado roaring': 249, 'people slurping': 250, 'dog baying': 251, 'motorboat, speedboat acceleration': 252, 'alarm clock ringing': 253, 'playing glockenspiel': 254, 'playing bass drum': 255, 'playing guiro': 256, 'heart sounds, heartbeat': 257, 'underwater bubbling': 258, 'playing steelpan': 259, 'people humming': 260, 'mouse pattering': 261, 'playing ukulele': 262, 'lathe spinning': 263, 'bull bellowing': 264, 'sailing': 265, 'crow cawing': 266, 'playing timpani': 267, 'sea waves': 268, 'chipmunk chirping': 269, 'playing bass guitar': 270, 'beat boxing': 271, 'female singing': 272, 'playing timbales': 273, 'ice cracking': 274, 'opening or closing car doors': 275, 'ferret dooking': 276, 'chicken clucking': 277, 'vehicle horn, car horn, honking': 278, 'car engine idling': 279, 'cutting hair with electric trimmers': 280, 'cat meowing': 281, 'playing cello': 282, 'playing trombone': 283, 'hammering nails': 284, 'playing clarinet': 285, 'playing squash': 286, 'playing saxophone': 287, 'people gargling': 288, 'playing bassoon': 289, 'wind rustling leaves': 290, 'hair dryer drying': 291, 'baby crying': 292, 'people slapping': 293, 'whale calling': 294, 'playing electronic organ': 295, 'people sobbing': 296, 'subway, metro, underground': 297, 'bird wings flapping': 298, 'woodpecker pecking tree': 299, 'cattle mooing': 300, 'ice cream truck, ice cream van': 301, 'writing on blackboard with chalk': 302, 'snake hissing': 303, 'baby laughter': 304, 'door slamming': 305, 'electric grinder grinding': 306, 'coyote howling': 307, 'wind chime': 308}
        class_counter = {key:0 for key in class_dict.keys()}
        self.class_dict = class_dict
        
        self.train_csv = os.path.join(self.data_root, args.dataset + '/train.csv')
        self.test_csv = os.path.join(self.data_root, args.dataset + '/test.csv')
        
        self.visual_feature_path =  Dataset_Config().get_dataset_root('VGGSound')
        self.audio_feature_path = os.path.join(Dataset_Config().get_dataset_root('VGGSound'), "preprocessed_AudioWAV")

        if mode in ['train', 'val']:
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
                    audio_path = os.path.join(self.audio_feature_path, 'v'+item[0] + '_out.pkl')
                elif self.audio_type == '.wav':
                    audio_path = os.path.join(self.audio_feature_path, 'v'+item[0] + '_out.wav')
                elif self.audio_type == '.npy':
                    audio_path = os.path.join(self.audio_feature_path, 'v'+item[0] + '_out.npy')
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(args.fps), 'v'+item[0])

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    if len(os.listdir(visual_path)) > 0:
                        self.image.append(visual_path)
                        self.audio.append(audio_path)
                        self.label.append(class_dict[item[1]])
                        class_counter[item[1]] += 1
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
