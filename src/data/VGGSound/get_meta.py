import pandas as pd
import os

def get_class(anns_path):
    df = pd.read_csv(anns_path, sep=',', header=None, names=['VideoID','StartTime', 'Category', 'split'])
    classes = []
    for index in range(len(df)):
        classes.append(df['Category'][index])
    classes = list(set(classes))
    classes = {classes[idx]:idx for idx in range(len(classes))}
    print(classes)
    
def get_meta(anns_path, split='train'):
    df = pd.read_csv(anns_path, sep=',', header=None, names=['VideoID','StartTime', 'Category', 'split'])
    classes = []
    my_meta_data = {
        'youtube_id': [],
        'label': []
    }
    for index in range(len(df)):
        if df['split'][index] == split:
            label = df['Category'][index]
            youtube_id = df['VideoID'][index]
            start_time = df['StartTime'][index]
            end_time = start_time + 10
            my_meta_data['youtube_id'].append("{}_{}_{}".format(youtube_id, start_time, end_time))
            my_meta_data['label'].append(label)
    my_meta_data = pd.DataFrame(my_meta_data)
    my_meta_data.to_csv(split+'.csv', index=False)
    
get_meta('vggsound.csv', 'train')
get_meta('vggsound.csv', 'test')
get_class('vggsound.csv')