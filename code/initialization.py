import cv2
import os
import shutil
import time
import numpy as np


FFMPEG_PREFIX = 'ffmpeg -i ./videos/'
FFMPEG_MIDDLE_1 = ' -ss '
FFMPEG_MIDDLE_2 = ' -to '
FFMPEG_SUFFIX = ' -c copy '

def split_videos(video_name, label_list, preparation=30, interval=5, manual=6):
    """
    This function split a video into several part and store them into the disk.
    :params: video_name: the name of video need to split
    :params: label_list: a list of strings containing labels of the output video clip
    :params: preparation: the preparation time in the video(s)
    :params: interval: interval time between each segment(s)
    :params: manual: according to Tom, the first 6 labels of the video are more credible.
    """
    assert manual <= 20, "The number of selected labels should less than or equal to 20"
    if not os.path.exists('./video_segments'): 
        os.makedirs('./video_segments')
    print('Start split video: ', video_name) 
    if manual != 0:
        label_list = label_list[:manual]
    num_segments = len(label_list)
    for i in np.arange(num_segments):
        seg_name = os.path.join('./video_segments/', 
                    video_name.split('.')[0] + str(i) + '_' + label_list[i] + '.avi')
        start_time = preparation
        end_time = preparation + 90
        os.system(FFMPEG_PREFIX + video_name + FFMPEG_MIDDLE_1 + str(start_time) + FFMPEG_MIDDLE_2 + str(end_time) + FFMPEG_SUFFIX + seg_name)
        preparation += (interval + 90)
    
    print('Split success, ', video_name)

def extract_frames(seg_path):
    if not os.path.exists('./dataset/train'): 
        os.makedirs('./dataset/train')
    video_list = os.listdir(seg_path)
    count =86
    i = 0
    j = 0
    for index, video_name in enumerate(video_list):
        counter = 0
        print('Start extracting frames of', video_name)
        label = video_name.split(".")[0].split('_')[-1]
        video_path = os.path.join('./video_segments/' + video_name)
        videoCapture = cv2.VideoCapture(video_path)
        frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = int(frame_count*0.5)
        interval = int(frame_count // 30)
        select_frames = np.linspace(0, frame_count-frame_count%30, interval).astype(int)
        while (counter < frame_count):
            success, frame = videoCapture.read()
            if counter in select_frames:
                counter += 1
                savedname = video_name.split(".")[0] + '_' + str(counter) + '_' + label + '.jpg'
                cv2.imwrite(os.path.join('./dataset/train/', savedname), frame)
                continue
            else:
                counter += 1
                continue
        videoCapture.release()
        time.sleep(5)

def split_train_test(data_path):
    if os.path.exists('./dataset/test'):
        print('Test set exists! Please check. If you wanna resplit the dataset, you need to extract frames first.')
        return -1
    else:
        print('Start split the datset')
        os.makedirs('./dataset/test')
        img_list = os.listdir(data_path)
        test_imgs = []
        nums = len(img_list)
        permutation = np.random.permutation(nums)
        test_idx = permutation[int(0.8*nums):]
        for idx in test_idx:
            test_imgs.append(img_list[idx])
        for test_img in test_imgs:
            try:
                shutil.move(data_path + '/' + test_img, './dataset/test/' + test_img)
            except Exception as e:
                print('move_file ERROR:', e)
        print('Split over.')    

def annotation(dir):
    imgs = os.listdir(dir)
    flag = dir.split('/')[-1]
    with open('./dataset/'+flag+'_annotation.txt', 'w') as file:
        for img in imgs:
            label = img.split('.')[0].split('_')[-1]
            if label == 'Stressful':
                file.write(img + ' 1\n')
            elif label == 'Calm':
                file.write(img + ' 0\n')
            else:
                print('Error:', img)
    print('Annoated.')