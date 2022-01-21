import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
from torchvision.utils import save_image

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 128

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 200

# Size of feature maps in generator
ngf = 128

# Size of feature maps in discriminator
ndf = 32

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1


# Number of gpus available
device = torch.device('cuda:0' if (
    torch.cuda.is_available() and ngpu > 0) else 'cpu')

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # input is Z, going into a convolution
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Create the generator
netG = Generator(ngpu).to(device)

### Generate bird image
# Apply the weights_init function to randomly initialize all weights
netG.load_state_dict(torch.load('./netGweightsBTattoo128v03'))

#real_cpu = data[0].to(device)
b_size = 1
noise = torch.randn(b_size, nz, 1, 1, device=device)
fake = netG(noise)

save_image(fake[0], 'bird.png')

### Generate scorpion image
# Apply the weights_init function to randomly initialize all weights
netG.load_state_dict(torch.load('./netGweightsScTattoo128v02'))

#real_cpu = data[0].to(device)
b_size = 1
noise = torch.randn(b_size, nz, 1, 1, device=device)
fake = netG(noise)

save_image(fake[0], 'scorpion.png')

### Generate skull image
# Apply the weights_init function to randomly initialize all weights
netG.load_state_dict(torch.load('./netGweightsSkTattoo128v02'))

#real_cpu = data[0].to(device)
b_size = 1
noise = torch.randn(b_size, nz, 1, 1, device=device)
fake = netG(noise)

save_image(fake[0], 'skull.png')





