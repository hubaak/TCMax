import pandas as pd
import os

splits = ['train', 'test', 'val']
dataset_dir = '/mnt/data_2/wuxy/datasets/AVE_Dataset'

def get_meta(anns_path, output_name):
    df = pd.read_csv(anns_path, sep='&', header=None, names=['Category', 'VideoID', 'Quality', 'StartTime', 'EndTime'])
    my_meta_data = {
        'youtube_id': [],
        'label': []
    }
    for index in range(len(df)):
        label = df['Category'][index]
        youtube_id = df['VideoID'][index]
        my_meta_data['youtube_id'].append(youtube_id)
        my_meta_data['label'].append(label)
    my_meta_data = pd.DataFrame(my_meta_data)
    my_meta_data.to_csv(output_name, index=False)
    
def get_class(anns_path):
    df = pd.read_csv(anns_path, sep='&', header=None, names=['Category', 'VideoID', 'Quality', 'StartTime', 'EndTime'])
    classes = []
    for index in range(len(df)):
        classes.append(df['Category'][index])
    classes = list(set(classes))
    classes = {classes[idx]:idx for idx in range(len(classes))}
    print(classes)

for split in splits:
    get_meta(os.path.join(dataset_dir, '{}Set.txt'.format(split)), '{}.csv'.format(split))

get_class(os.path.join(dataset_dir, 'trainSet.txt'))