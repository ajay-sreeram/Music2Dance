from custom_utils.video_shot_processing_utils import PersonDetection, PoseDetection
from custom_utils.datastft import single_spectrogram
import numpy as np
import matplotlib.pyplot as plt

import cv2
import math
from scipy.io import wavfile

import pylab
import json
import sys
import pickle

import tensorflow as tf


# right now not usefull because different images has different size boxes of person detected

def _get_cropped_images_and_pos(images_np):
    detector = PersonDetection()
    pose = PoseDetection()
    (images_np_cropped, isperson_dected) = detector.get_person(images_np)
    (images_pose_points, pose_scores) = pose.get_pose(images_np_cropped)
    return (images_pose_points, images_np_cropped, isperson_dected)

def _get_cropped_images(images_np):
    detector = PersonDetection()
    at_a_time_images = 500
    #(images_np_cropped, isperson_dected) = detector.get_person(images_np)
    (images_np_cropped, isperson_dected) = detector.get_person(images_np[:at_a_time_images])
    batches = math.ceil(len(images_np)/at_a_time_images)
    for i in range(1, batches):
        (images_np_cropped_new, isperson_dected_new) = detector.get_person(images_np[i*at_a_time_images:(i+1)*at_a_time_images])
        images_np_cropped = np.concatenate((np.array(images_np_cropped), np.array(images_np_cropped_new)), 0)
        isperson_dected = np.concatenate((isperson_dected, isperson_dected_new), 0)
    return (images_np_cropped, isperson_dected)


def points_to_ratios(points, image_cropped):
    #change dtype to float so that in below steps it will get round of to zero
    points = points.astype(float)
    
    #point y-axis coord / height
    points[0][:,0] = points[0][:,0]/image_cropped.shape[0] 
    #point x-axis coord / width
    points[0][:,1] = points[0][:,1]/image_cropped.shape[1]

    return points

images_pose_index = 0

def _get_cropped_images_pos(images_np, detector=None, pose=None):
    global images_pose_index
    if detector is None:
        detector = PersonDetection()
    is_new_session = False
    if pose is None:
        pose = PoseDetection()
        is_new_session = True
    at_a_time_images = 800
    #(images_np_cropped, isperson_dected) = detector.get_person(images_np)
    (images_np_cropped, isperson_dected) = detector.get_person(images_np[:at_a_time_images])
    images_pose_points = dict()
    
    for index in range(len(isperson_dected)):
        if isperson_dected[index]:
            #images_pose_points[images_pose_index] = pose.get_pose(np.expand_dims(images_np_cropped[index], 0))
            (images_pose, pose_scores) = pose.get_pose(np.expand_dims(images_np_cropped[index], 0))
            images_pose = points_to_ratios(images_pose, images_np_cropped[index])
            images_pose_points[images_pose_index] = (images_pose, pose_scores)
        images_pose_index = images_pose_index+1
    
    batches = math.ceil(len(images_np)/at_a_time_images)
    for i in range(1, batches):
        (images_np_cropped_new, isperson_dected_new) = detector.get_person(images_np[i*at_a_time_images:(i+1)*at_a_time_images])
        #images_np_cropped = np.concatenate((images_np_cropped, images_np_cropped_new), 0)
        isperson_dected = np.concatenate((isperson_dected, isperson_dected_new), 0)
        for index in range(len(isperson_dected_new)):
            if isperson_dected_new[index]:
                #images_pose_points[images_pose_index] = pose.get_pose(np.expand_dims(images_np_cropped_new[index], 0))
                (images_pose, pose_scores) = pose.get_pose(np.expand_dims(images_np_cropped_new[index], 0))
                images_pose = points_to_ratios(images_pose, images_np_cropped_new[index])
                images_pose_points[images_pose_index] = (images_pose, pose_scores)
            images_pose_index = images_pose_index+1
    
    if is_new_session:
        pose.pose_session.close()
    return (images_pose_points, isperson_dected)

def process_video_with_index(video_index, enable_spectrogram = True):
    global images_pose_index
    images_pose_index = 0
    min_avg_pose_score = .8
    vidcap = cv2.VideoCapture('data/video_wav_indexed/video ({0}).mp4'.format(video_index))
    audio_rate, audio = wavfile.read('data/audio_wav_indexed/audio ({0}).wav'.format(video_index))
    # if audio other than mono take 1st channel
    if (len(audio.shape)>1):
        audio = audio[:,0]
    num_secs = math.floor(len(audio)/audio_rate)
    video_images = []

    if enable_spectrogram:
        next_video_frame_dist = 1000
    else:
        next_video_frame_dist = 250 #make it 333 (int(1000/3)) because video frame rate is 333 (1000/3)
        num_secs = num_secs*3
        audio_rate = math.floor(audio_rate/3)

    images_poses = dict()
    isperson_dected = np.array([])
    for sec in range(1, num_secs+1):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,next_video_frame_dist*sec)
        success, image = vidcap.read()
        if success == False:
            break
        video_images.append(image)
        if len(video_images) == 100:
            (images_poses_new, isperson_dected_new) = _get_cropped_images_pos(np.array(video_images))
            images_poses.update(images_poses_new)
            isperson_dected = np.concatenate((isperson_dected, isperson_dected_new), 0)
            video_images = []
            print("finished screens: {0}".format(sec))
        
    if len(video_images)!=0:
        (images_poses_new, isperson_dected_new) = _get_cropped_images_pos(np.array(video_images))
        images_poses.update(images_poses_new)
        isperson_dected = np.concatenate((isperson_dected, isperson_dected_new), 0)
        video_images = []
    
    #video_images = np.array(video_images)
    #(images_poses, isperson_dected) = _get_cropped_images_pos(video_images)
    #(images_cropped, isperson_dected) = _get_cropped_images(video_images)
    #del(video_images)
    sys.stdout.write("finished getting person boxes\n")
    processed_data_map = []
    #pose = PoseDetection()

    partEdges = [
        [5, 6], [5,7], [7,9], [6,8], [8,10], [5,11], [6,12], 
        [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
    ]

    for index in range(len(isperson_dected)):
        if isperson_dected[index]:
            #pose points (replace 0 with index if calling _get_cropped_images_and_pos fun)
            (images_pose_points, pose_scores) = images_poses[index]
            if pose_scores[0] < min_avg_pose_score:
                continue
                        
            '''
            below steps done in previous function
            #change dtype to float so that in below steps it will get round of to zero
            images_pose_points = images_pose_points.astype(float)
            
            #point y-axis coord / height
            images_pose_points[0][:,0] = images_pose_points[0][:,0]/images_cropped[index].shape[0] 
            #point x-axis coord / width
            images_pose_points[0][:,1] = images_pose_points[0][:,1]/images_cropped[index].shape[1]             
            
            #cropped image
            #cv2.imwrite("data/processed/video_{0}_screen_{1}_frame.jpg".format(video_index, index), images_cropped[index])
            '''
            if enable_spectrogram:
                #spectogram
                pylab.ioff()
                pylab.figure(num=None, figsize=(3,2.5))
                pylab.axis('off')
                pylab.specgram(audio[index * audio_rate : (index+1) * audio_rate], Fs=audio_rate)
                pylab.savefig('data/processed/video_{0}_screen_{1}_spectogram.png'.format(video_index, index), 
                                transparent = True, bbox_inches = 'tight', pad_inches = 0)
                pylab.close()            
                processed_data_map.append({
                    'points': np.squeeze(images_pose_points[0].reshape((-1,1)),1).tolist(),
                    'video_index':video_index,
                    'screen_index':index,
                    'pose_score':pose_scores[0]
                })
            else:
                #sample frame rate = 48000 but taking 3 shots in each so 48000/3=16000
                audio_samples = np.zeros([16000]).astype(int)
                audio_samples[:audio_rate] = audio[index * audio_rate : (index+1) * audio_rate]
                processed_data_map.append({
                    'points': np.squeeze(images_pose_points[0].reshape((-1,1)),1).tolist(),
                    'video_index':video_index,
                    'screen_index':index,
                    'pose_score':pose_scores[0],
                    'audio_samples':audio_samples.tolist()
                })                


    sys.stdout.write("saving processed dict for video:{0}\n".format(video_index))
    with open('data/processed/video_{0}_preprocessed.pickle'.format(video_index), 'wb') as outfile:
        pickle.dump({'preprocessed':processed_data_map}, outfile, pickle.HIGHEST_PROTOCOL)
    
    sys.stdout.write("saving processed json for video:{0}\n".format(video_index))
    with open('data/processed/video_{0}_preprocessed.json'.format(video_index), 'w') as outfile:
        json.dump({'preprocessed':processed_data_map}, outfile)
    
    return processed_data_map


# this function is like iterator, uses less memory storage
def process_video_with_index_less_memory(video_index):
    global images_pose_index
    images_pose_index = 0
    # keeping .7 only because to get more training data for pose completion
    min_avg_pose_score = .5
    vidcap = cv2.VideoCapture('data/video_wav_indexed/video ({0}).mp4'.format(video_index))
    audio_rate, audio = wavfile.read('data/audio_wav_indexed/audio ({0}).wav'.format(video_index))
    # if audio other than mono take 1st channel
    if (len(audio.shape)>1):
        audio = audio[:,0]
    num_secs = math.floor(len(audio)/audio_rate)
    video_images = []
    # de allocate audio memory
    #audio = None
    
    # because i need number of frames per second are 30
    fps = 30
    next_video_frame_dist = int(1000/fps) 
    num_secs = num_secs*fps
    process_batch_len = fps*3 # simple taking 3 times of fps
    actual_audio_rate = audio_rate
    #actual_audio_len = len(audio)
    audio_rate = int(audio_rate/30)

        
    pose_result = []

    detector = PersonDetection()
    pose = PoseDetection()
    sys.stdout.write("TOTAL SECS: {0}\n".format(num_secs))
    for sec in range(1, num_secs+1):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,next_video_frame_dist*sec)
        success, image = vidcap.read()
        if success == False:
            break
        video_images.append(image)
        if sec%10==0:
            sys.stdout.write("\r\r :: finished: {0}%, secs: {1}".format(int((sec/num_secs)*100), sec))
        if len(video_images) == process_batch_len:
            (images_poses, _) = _get_cropped_images_pos(np.array(video_images), detector, pose)
            for index in images_poses.keys():
                image_pose, score = images_poses[index]
                score = score[0]
                if score > min_avg_pose_score:
                    audio_index = [audio_rate*index, audio_rate*(index+1)]
                    pose_result.append({
                        'pose_points':np.squeeze(image_pose[0].reshape((-1,1)),1).tolist(),
                        'pose_score':score,
                        'screen_position':next_video_frame_dist*(index+1),
                        'audio_position':audio_index,
                        'audio_spec':_get_stft_spectogram(audio[audio_index[0]:audio_index[1]], actual_audio_rate)
                    })
                    #not saving raw audio above because, we can fetch it from audio based on above audio_position in below actual_video_index audio file
            video_images = []  
        
    sys.stdout.write("\nFound poses in video {0}: {1}\n".format(video_index, len(pose_result)))
    video_images = []
    pose.pose_session.close()
    
    sys.stdout.write("saving processed json for video:{0}\n".format(video_index))
    with open('data/processed_30fps/video_{0}_preprocessed.json'.format(video_index), 'w') as outfile:
        json.dump({'preprocessed':pose_result,
                    'fps': fps,
                    'actual_audio_rate':actual_audio_rate,
                    'updated_audio_rate':audio_rate,
                    'actual_num_secs':math.floor(len(audio)/actual_audio_rate),
                    'updated_num_secs':num_secs,
                    'total_obtained_samples':len(pose_result),
                    'next_video_frame_dist':next_video_frame_dist,
                    'actual_video_index':video_index
                    }, outfile)

    return 


def _get_stft_spectogram(wav_raw, audio_rate):
    '''
    copied below values from authors code
    audio_max = 5.
    audio_min = -120.
    rng_wav_min = -0.9
    rng_wav_max = 0.9
    slope_wav = (rng_wav_max - rng_wav_min) / (audio_max - audio_min)
    intersec_wav = rng_wav_max - slope_wav * audio_max
    
    as authors sample rate is 16000 and wlen = 160, hop = 80 result shape is 129,5
    
    so if audio_rate == 48000
    freq = audio_rate
    wlen = 160*4 = 640
    hop = 80*3 = 240
    then result shape is 513,5

    if audio_rate = 44100
    freq = audio_rate
    wlen = 160*4 = 640
    hop = 80*2.5 = 200
    
    with this calc below values
    '''
    slope_wav = 0.0144
    intersec_wav = 0.8280000000000001
    freq = audio_rate
    wlen = 640
    if audio_rate == 48000:
        hop = 240
    elif audio_rate == 44100:
        hop = 200
    else:
        raise Exception('Invalid sample rate {0}'.format(audio_rate))    

    stft_data = single_spectrogram(wav_raw, freq, wlen, hop) * slope_wav + intersec_wav
    return np.swapaxes(stft_data, 0, 1).tolist()

#from video_shot_processing import process_video_with_index, process_video_with_index_less_memory

def main():    
    num_videos = 36
    tf.logging.set_verbosity(tf.logging.ERROR)
    for video_index in range(1, num_videos+1):
        sys.stdout.write('processing video:{0}\n'.format(video_index))
        process_video_with_index_less_memory(video_index)

if __name__ == '__main__':
    main()