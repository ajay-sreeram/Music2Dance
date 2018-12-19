
# coding: utf-8

# In[ ]:


#to create processed_30fps_only_inout using processed_30fps
#works perfectly well


# In[1]:


import os
import numpy as np
import json
import sys
import matplotlib.pyplot as plt


# In[2]:


input_meta_info = {
    'input_audio_spect':[],
    'output_pose_point':[]
}
output_path = "data/processed_30fps_only_inout"
g_min = 0
g_bundle_size = 1000
g_total = 0


# In[10]:


def process_one_file_content(file_content):
    global input_meta_info, g_total, g_min, g_bundle_size, output_path, tmp_file_content
    min_pose_score = .65
    for content in file_content:
        if content['pose_score'] > min_pose_score:
            input_meta_info['input_audio_spect'].append( np.array(content['audio_spec']) )
            input_meta_info['output_pose_point'].append( np.array(content['pose_points']) )
            while len(input_meta_info['input_audio_spect']) == g_bundle_size:
                input_meta_info['input_audio_spect'] = np.array(input_meta_info['input_audio_spect']).tolist()
                input_meta_info['output_pose_point'] = np.array(input_meta_info['output_pose_point']).tolist()
                input_meta_info['g_min'] = g_min
                cur_out_file = '/bundle_{0:09d}_{1:09d}.json'.format(g_min, g_min+g_bundle_size)                
                sys.stdout.write("\r\rbundle: {0} ".format(cur_out_file))
                g_min = g_min + g_bundle_size
                g_total = g_total + g_bundle_size                
                with open(output_path+cur_out_file, 'w') as outfile:
                    json.dump(input_meta_info, outfile)
                    input_meta_info = {
                        'input_audio_spect':[],
                        'output_pose_point':[]
                    }               
    
def process_files(folder):
    #"data/processed_30fps"
    files_cnt = 0
    for root, _, files in os.walk(folder):#processed_30fps #tmp
        for file in files:
            if file.endswith(".json"):
                files_cnt = files_cnt + 1
                sys.stdout.write("\nprocessing file: {0} - {1}\n".format(files_cnt, file))
                with open(os.path.join(root, file)) as f:
                    process_one_file_content( np.array(json.load(f)['preprocessed']) )                
                #if files_cnt == 3:
                #    break


# In[11]:


if not os.path.exists(output_path):
    os.makedirs(output_path)

process_files("data/processed_30fps")

print("\nfinished processing total samples_found: {0}".format(g_total))
with open(output_path+"/total_samples_cnt.txt", 'w') as outfile:
    json.dump({
        'total_samples_cnt':g_total
        }, outfile)                
