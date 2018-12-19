
# coding: utf-8

# In[1]:


#get_ipython().system('sudo pip install imageio')
#get_ipython().system('sudo rm -rf ~/.local/share/Trash/*')


# In[2]:


import torch

if torch.cuda.device_count()>0:
	torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
	torch.set_default_tensor_type('torch.DoubleTensor')

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import json
import sys
import numpy as np
import math
import time
import imageio

from tensorboardX import SummaryWriter

from custom_utils.datastft import single_spectrogram

import logging


# In[3]:


class PoseMusicDataset_new(Dataset):

    def __init__(self, bundle_len = 1000, seq_len = 100, init_step = 0,  prev_poses_cnt = 5):
        self.prev_poses_cnt = prev_poses_cnt
        
        self.last_n_poses = None
        
        #self.max_raw_sample = 175000
        self.seq_len = seq_len
        self.bundle_len = bundle_len
        self.data_files_path = "data/processed_30fps_only_inout"
        self.seq_samples_cnt = 790 #1750 
        # because max_raw_sample/seq_len = 1750 ie., each bundle has 10 i/o for this model as seq_len is 100
        
        self.seq_samples_ids_stack = []
        self.seq_samples_db = dict()
        self.max_db_size = 50 
        # better to have multiples of 10
        
    def __len__(self):
        return self.seq_samples_cnt

    def __getitem__(self, idx):
        bundle_min = int(math.floor(idx/10))*self.bundle_len
        index_in_bundle = int(idx%10)*self.seq_len
        sample_key = "{0}_{1}".format(bundle_min, index_in_bundle)
        
        #check if in db
        if sample_key in self.seq_samples_db:
            return self.seq_samples_db[sample_key]        
        
        if len(self.seq_samples_db) >= self.max_db_size:            
            for index in range(10):
                del(self.seq_samples_db[self.seq_samples_ids_stack[index]])
            self.seq_samples_ids_stack = np.array(self.seq_samples_ids_stack)[10:].tolist()
            p_time = time.time()
        
        #add corresponding index file samples, these will be 10 always
        with open(self.data_files_path+"/bundle_{0:09d}_{1:09d}.json".format(bundle_min, bundle_min+self.bundle_len)) as f:
            p_time = time.time()
            file_data = json.load(f)
            input_audio_spect = np.array(file_data['input_audio_spect'])
            output_pose_point = np.array(file_data['output_pose_point'])
            del(file_data)
            for index in range(10):
                cur_index_in_bundle = index*self.seq_len
                audio_inputs = input_audio_spect[cur_index_in_bundle:cur_index_in_bundle+self.seq_len]
                audio_inputs = np.expand_dims(audio_inputs, 1)
                next_steps = output_pose_point[cur_index_in_bundle:cur_index_in_bundle+self.seq_len]
                
                if self.last_n_poses is None:
                    prev_poses_input = np.zeros((self.prev_poses_cnt, next_steps.shape[-1]))#, dtype=np.float32)
                    prev_poses_target = np.zeros((self.prev_poses_cnt, next_steps.shape[-1]))#, dtype=np.float32)
                else:
                    prev_poses_input = self.last_n_poses[:-1]
                    prev_poses_target = self.last_n_poses[1:]
                                
                sample = {
                    'audio_inputs': audio_inputs,
                    'prev_poses': prev_poses_input,  #np.zeros(34, dtype=np.float32),
                    'next_steps': np.concatenate((prev_poses_target, next_steps), 0)
                }
                cur_sample_key = "{0}_{1}".format(bundle_min, cur_index_in_bundle)
                self.seq_samples_ids_stack.append(cur_sample_key)
                self.seq_samples_db[cur_sample_key] = sample
                self.last_n_poses = sample['next_steps'][-(self.prev_poses_cnt+1):]
        
        return self.seq_samples_db[sample_key]


# In[4]:


seq_len = 100
prev_poses_cnt = 5
dset = PoseMusicDataset_new(1000, seq_len, 0, prev_poses_cnt)
epochs = 500

batch_size = 10
dataloader = DataLoader(dset, batch_size=batch_size,shuffle=False, num_workers=0)
print("Epochs to do:", epochs)
print("Dataloader size:", dataloader.__len__())
print("Dataset size:", dset.__len__())


# In[5]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states] 


# In[6]:


output_dir = "output/motiondance_simplernn"
if not os.path.exists(output_dir+"/checkpoints"):
    os.makedirs(output_dir+"/checkpoints")
if not os.path.exists(output_dir+"/frozen"):
    os.makedirs(output_dir+"/frozen")

logging.basicConfig(format='%(asctime)s :%(message)s', 
                    level=logging.INFO,
                    filename='./{0}/{1}.log'.format(output_dir, 'info')
                    )
logging.info("Training Started:")
logging.info("----{0}----".format(0))
sys.stdout.write("created required directories for saving model\n")


# In[7]:


writer = SummaryWriter(comment='mdn_simple_rnn_gpu_{0}'.format(epochs))


# In[8]:


class CNNFeat(torch.nn.Module):
    def __init__(self, dim):
        super(CNNFeat, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=(129, 2))
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=(129, 2))
        self.conv3 = torch.nn.Conv2d(16, 24, kernel_size=(129, 2))
        self.conv4 = torch.nn.Conv2d(24, dim, kernel_size=(129, 2))
        self.cvbn1 = torch.nn.BatchNorm2d(8)
        self.cvbn2 = torch.nn.BatchNorm2d(16)
        self.cvbn3 = torch.nn.BatchNorm2d(24)
        self.cvbn4 = torch.nn.BatchNorm2d(dim)
        
    def forward(self, h):
        h = F.elu(self.cvbn1(self.conv1(h)))
        h = F.elu(self.cvbn2(self.conv2(h)))
        h = F.elu(self.cvbn3(self.conv3(h)))
        h = F.elu(self.cvbn4(self.conv4(h)))
        return h.view((h.size(0), -1))


# In[9]:


class MDNRNN(torch.nn.Module):
    def __init__(self, dim, cnnEncoder, z_size, n_hidden=256, n_gaussians=5, n_layers=1):
        super(MDNRNN, self).__init__()
        
        self.z_size = z_size
        self.n_hidden = n_hidden
        self.n_gaussians = n_gaussians
        self.n_layers = n_layers
        
        self.lstm = torch.nn.LSTM(dim, n_hidden, n_layers, batch_first=True)
        self.prev_steps_fc = torch.nn.Linear(z_size, dim)
        self.audiofeat = cnnEncoder(dim)        
        self.fc1 = torch.nn.Linear(n_hidden, n_gaussians)#*z_size)
        self.fc2 = torch.nn.Linear(n_hidden, n_gaussians*z_size)
        self.fc3 = torch.nn.Linear(n_hidden, n_gaussians)#*z_size)
        
    def get_mixture_coef(self, y):
        rollout_length = y.size(1)
        pi, mu, sigma = self.fc1(y), self.fc2(y), self.fc3(y)
        
        pi = pi.view(-1, rollout_length, self.n_gaussians)
        mu = mu.view(-1, rollout_length, self.z_size, self.n_gaussians)
        sigma = sigma.view(-1, rollout_length, self.n_gaussians)#, self.z_size)
        
        pi = F.softmax(torch.clamp(pi, 1e-8, 1.), -1)
        
        sigma = F.elu(sigma)+1.+1e-8
        return pi, mu, sigma
        
        
    def forward(self, audio_inputs, prev_steps, h):
        # Forward propagate LSTM
        x = []
        
        for i, input_t in enumerate(prev_steps.chunk(prev_steps.size(1), dim=1)):
            p_steps = self.prev_steps_fc(input_t)
            x += [p_steps.view((p_steps.size(0), -1))]
            
        for i, input_t in enumerate(audio_inputs.chunk(audio_inputs.size(1), dim=1)):
            input_t = input_t[:,0]
            h_ = self.audiofeat(input_t)
            x += [h_]
        
        x = torch.stack(x, 1).squeeze(2)
        y, (h, c) = self.lstm(x, h)
        pi, mu, sigma = self.get_mixture_coef(y)
        return (pi, mu, sigma), (h, c)
    
    def init_hidden(self, bsz):
        return (torch.zeros(self.n_layers, bsz, self.n_hidden).to(device),
                torch.zeros(self.n_layers, bsz, self.n_hidden).to(device))


# In[10]:


#reference https://github.com/axelbrando/Mixture-Density-Networks-for-distribution-and-uncertainty-estimation/blob/master/MDN-3D-Regression.ipynb
#https://github.com/sksq96/pytorch-mdn/blob/master/mdn-rnn.ipynb
def log_sum_exp(x, dim=None):
    """Log-sum-exp trick implementation"""
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_log = torch.log(torch.sum(torch.exp(x - x_max), dim=dim, keepdim=True))
    return x_log+x_max
        
def mdn_loss_fn(y, pi, mu, sigma):    
    c = y.shape[-2]
    
    var = (sigma ** 2)
    log_scale = torch.log(sigma)    
    
    exponent = torch.log(pi) - .5 * float(c) * math.log(2 * math.pi)         - float(c) * log_scale         - torch.sum(((y - mu) ** 2), dim=2) / (2 * var)
    
    log_gauss = log_sum_exp(exponent, dim=2)
    res = - torch.mean(log_gauss)

    return res

def criterion(y, pi, mu, sigma):
    y = y.unsqueeze(3)
    return mdn_loss_fn(y, pi, mu, sigma)

def get_predicted_steps(pi, mu):
    pi = pi.cpu().detach().numpy()
    dim = pi.shape[2]
    z_next_pred = np.array([ [mu[i,seq,:,np.random.choice(dim,p=pi[i][seq])].cpu().detach().numpy() for seq in np.arange(pi.shape[1])] for i in np.arange(len(pi))])
    return z_next_pred


# In[11]:


def save_checkpoint(save_path, epoch, model, optimizer):
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, "{0}/epoch_300_plus_{1}.pth.tar".format(save_path, epoch+1))

def load_checkpoint(model, optimizer, save_path):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(save_path):
        print("=> loading checkpoint '{}'".format(save_path))
        checkpoint = torch.load(save_path)#, map_location=lambda storage, loc: storage)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})" .format(save_path, checkpoint['epoch']))
        
        model = model.to(device)
        # now individually transfer the optimizer parts...
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return start_epoch, model, optimizer


# In[12]:


audio_convout_size = 28
z_size = 34 #output size
n_hidden = 512
n_gaussians = 5
n_layers = 2

gpu_cnt = torch.cuda.device_count()
if gpu_cnt == 1:
    sys.stdout.write("One GPU\n")
    model = MDNRNN(audio_convout_size, CNNFeat, z_size, n_hidden, n_gaussians, n_layers).cuda()
elif gpu_cnt > 1:
    sys.stdout.write("More GPU's: {0}\n".format(gpu_cnt))
    model = torch.nn.DataParellel( MDNRNN(audio_convout_size, CNNFeat, z_size, n_hidden, n_gaussians, n_layers).cuda() )
else:
    sys.stdout.write("No GPU\n")
    model = MDNRNN(audio_convout_size, CNNFeat, z_size, n_hidden, n_gaussians, n_layers)
    
model = model.double()
    
#criterion = torch.nn.MSELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999), amsgrad=True)
optimizer = torch.optim.Adam(model.parameters())

partEdges = [
        [5, 6], [5,7], [7,9], [6,8], [8,10], [5,11], [6,12], 
        [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]
    ]
plot_image_size = 20
g_p_time = time.time()
p_time = time.time()


# In[13]:

'''
# to continue from previous checkpoint
model_saved_path = "output/motiondance_simplernn_100To300/checkpoints/epoch_100_plus_300.pth.tar"
_, model, optimizer = load_checkpoint(model, optimizer, model_saved_path)
'''

projection_print_index = len(dataloader)#dataloader.__len__()*5

model = model.train()
for epoch in range(epochs):
    total_loss = 0.0
    hidden = model.init_hidden(batch_size)
    for i, data in enumerate(dataloader):
        if gpu_cnt>0:
            audio_inputs = data['audio_inputs'].type(torch.cuda.DoubleTensor)
            prev_steps = data['prev_poses'].type(torch.cuda.DoubleTensor)
            next_steps = data['next_steps'].type(torch.cuda.DoubleTensor)    
        else:
            audio_inputs = data['audio_inputs'].type(torch.DoubleTensor)
            prev_steps = data['prev_poses'].type(torch.DoubleTensor)
            next_steps = data['next_steps'].type(torch.DoubleTensor)    
                
        optimizer.zero_grad()
        
        hidden = detach(hidden)
        hidden = model.init_hidden(batch_size)
        
        (pi, mu, sigma), hidden = model(audio_inputs, prev_steps, hidden)        
        
        loss = criterion(next_steps, pi, mu, sigma)                
                
        loss.backward()
        optimizer.step()
                
        cur_loss = loss.cpu().detach().numpy()
        total_loss += cur_loss                        
        
        n_iter = epoch*len(dataloader) + i
        writer.add_scalar('cur_loss', cur_loss, n_iter+1)
        writer.add_scalar('loss', total_loss/(i+1), n_iter+1)
        
        sys.stdout.write('\r\r\r[{:8d}, {:3d}, {:5d}] tot_loss: {:12.6f} cur_loss: {:12.6f}  tot_time: {:17.4f}'.format(n_iter+1, epoch + 1, i + 1, total_loss/(i+1), cur_loss, (time.time()-g_p_time) ))
        logging.info('[{:8d}, {:3d}, {:5d}] tot_loss: {:12.6f} cur_loss: {:12.6f} tot_time: {:17.4f}'.format(n_iter+1, epoch + 1, i + 1, total_loss/(i+1), cur_loss, (time.time()-g_p_time) ))                        
    
    logging.info('epoch {0:3d} finished'.format(epoch+1))
    sys.stdout.write('\nepoch {0:3d} finished\n'.format(epoch+1))
    writer.add_scalar('epoch_loss', total_loss/(i+1), epoch+1)
    if(epoch+1)%10 == 0:
        save_checkpoint(output_dir+"/checkpoints", epoch, model, optimizer)
        sys.stdout.write("\ncheckpoint saved for epoch+1: {0}\n".format(epoch+1))
        logging.info("checkpoint saved for epoch+1: {0}".format(epoch+1))
                
sys.stdout.write("\n---- Finished processing ----\n")


# In[ ]:


final_freezed_path = output_dir+"/frozen/final_model_{0}.pt".format(epochs)
torch.save(model, final_freezed_path)

sys.stdout.write("---- Model Frozen ----\n")
logging.info("---- Model Frozen ----")

sys.stdout.write("\n---- DONE ----\n")