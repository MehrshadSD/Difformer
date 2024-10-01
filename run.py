# %% [markdown]
# ### Spatially-Aware Just-in-Time Autoregressive Diffusion
# 
# Convert an image into patches
# 
# Apply noise jointly across spatial and temporal domains onto patches
# 
# Predict the noise 
# 
# Lock in cell in the center and save that as the image
# 
# 150 step diffusion
# 
# 150   1        150
# 
# past  current  future
# 
# Technically this is doing two step attending - one attending only for past - creating cond sequence
# 
# cond sequence then fed into future attending to generate diffusion results
# 
# Autoregressive Diffusion inference speed sped up from O(nm) to O(n+m)!!!
# 
# # TODO
# 
# Implement EMA
# Double check parts deviating from Lucid's implementation
# - difformer line 293 to 310
# Implement position embedding / absolute position embedding
# - Sinusiodal embeddings need to be updated to work with multi-timestep logic
# 
# 

# %%
from PIL import Image
import numpy as np
import scipy.io
import gc
from tqdm import tqdm
import numpy as np

PATCH_SIZE = 8
SAMPLE_STEPS = 256
WINDOW_SIZE = SAMPLE_STEPS
SAMPLE_SIZE = 625
data_path = "../data/jpg/image_00001.jpg"
label_path = "../data/jpg/imagelabels.mat"
device = "cuda"

with Image.open(data_path) as im:
    a = np.asarray(im)
    print(a.shape)
    b = Image.fromarray(a, mode="RGB")
    b.save("./test.jpg")


# %%
# import os, sys

# path = "../data/jpg/"
# dirs = os.listdir( path )

# def resize():
#     for item in dirs:
#         if ".jpg" in item and "resized" not in item:
#             if os.path.isfile(path+item):
#                 im = Image.open(path+item)
#                 f, e = os.path.splitext(path+item)
#                 imResize = im.resize((200,200), Image.LANCZOS)
#                 imResize.save(f + '_resized.jpg', 'JPEG', quality=90)

# resize()

# %%
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

def img_norm(img):
    return img / 255

def img_crop(img, patch_size):
    height= (img.shape[0]//patch_size)*patch_size
    width = (img.shape[1]//patch_size)*patch_size
    plen = (img.shape[0]//patch_size) * (img.shape[1]//patch_size)
    return img[:height, :width, :], plen

def get_dataset(root, label_path, patch_size, sample_steps):
    labels = scipy.io.loadmat(label_path)
    labels = labels['labels'][0]
    dataset = []
    l = []
    for i, idx in enumerate(tqdm(labels)):
        fp = root +"image_"+str(i+1).rjust(5,'0')+"_resized.jpg"
        f = open(fp, 'rb')
        image = Image.open(f)
        image, plen = img_crop(np.array(image), patch_size)
        l.append(plen)
        mask = [0] * (plen+sample_steps)
        mask[1:plen+1] = [i+3 for i in range(plen)]
        mask[0] = 1
        mask[plen] = 2
        mask = np.pad(mask, (sample_steps,0), mode="constant", constant_values=0)

        dataset.append(
            {
                'img':image,
                'label':idx,
                'mask':mask,
                'plen':plen
            }
        )
        del mask
        del image

    print(max(l))
    return dataset

# Oxford flowers dataset 
class FlowerDataset(Dataset):
    def __init__(self,
                 patch_size = 8,
                 sample_steps = 99,
                 label_path = "../data/jpg/imagelabels.mat", 
                 root = "../data/jpg/"):
        self.dataset = get_dataset(root, label_path, patch_size, sample_steps)
        self.sample_steps = sample_steps

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        plen = self.dataset[idx]['plen']
        offset = np.random.randint(self.sample_steps+1,  self.sample_steps + plen-1)
        return {
            'img': self.dataset[idx]['img'],
            'mask': self.dataset[idx]['mask'],
            'label': self.dataset[idx]['label'] - 1,
            'offset':offset
        }
    
trainset = FlowerDataset(patch_size = PATCH_SIZE , sample_steps = SAMPLE_STEPS)
trainloader = DataLoader(trainset, batch_size=48) 

# %%
from difformer import ArSpImageDiffusion
import torch

model = ArSpImageDiffusion(
    model = dict(
        dim = 1024,
    ),
    patch_size = PATCH_SIZE,
    num_classes = 102,
    window_size = WINDOW_SIZE,
    sample_steps = SAMPLE_STEPS,
    sample_size = SAMPLE_SIZE
)
model.to(device)


# %%
from einops import rearrange, repeat, reduce, pack, unpack

sigmas = model.model.diffusion.sample_schedule()
spatial = model.model.diffusion.sample_spatial(80)

sigma = repeat(sigmas[spatial], "d -> b d 1", b = 1)

print(sigma.shape)


# gammas = torch.where(
#     (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
#     min(self.S_churn / sample_steps, sqrt(2) - 1),
#     0.
# )


# sigma = repeat(sigmas[spatial], "d -> b d 1", b = shape[0])
# gamma = repeat(gammas[spatial], "d -> b d 1", b = shape[0])
# sigma_next = sigma - 1

# %%
def train(model, dataloader, optimizer):
    model.train()
    running_loss = 0
    total_steps = 0
    for i, b in enumerate(tqdm(dataloader)):
        img = b['img'].to(device).float()
        mask = b['mask'].to(device).int()
        label = b['label'].to(device).int()
        offset = b['offset'].int()

        optimizer.zero_grad()
        loss = model(img, mask, offset, label)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_steps += 1
    return running_loss/total_steps

# %%
def inference(model, num_images):
    model.eval()
    for l in range(10):
        for j in range(num_images):
            sampled = model.sample(batch_size = 1, label=torch.tensor(l).to(device))
            img = Image.fromarray(sampled.squeeze().cpu().numpy().astype(np.uint8), mode='RGB')
            img.save("./results/"+str(l)+"_"+str(j)+".jpg")
        break


# %%
import torch.optim as optim

epochs = 500
optimizer = optim.Adam(model.parameters(), lr=0.001)
for e in range(epochs):
    loss = train(model, trainloader, optimizer)
    print(e, " avg loss:{:.3f}".format(loss))

    if e%10 == 0 and e>0:
        inference(model, 1)


