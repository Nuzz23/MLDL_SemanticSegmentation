from torch import Tensor
import torch.nn as nn


class FCDiscriminator(nn.Module):
    def __init__(self, num_classes:int=19, ndf:int=64)->None:
        """
        Fully Convolutional Discriminator for semantic segmentation tasks.
        
        Args:
            num_classes (int): Number of classes in the segmentation task.
            ndf (int): Number of filters in the first convolutional layer.
        """

        super(FCDiscriminator, self).__init__()

        # Recieves as input a tensor of shape Bx3x1280x720 where B is the batch size, 3 is the number of channels (RGB),
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1) # Bx 64 x 640 x 360
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1) # Bx 128 x 320 x 180
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1) # Bx 256 x 160 x 90
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1) # Bx 512 x 80 x 45
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1) # Bx 1 x 39 x 22

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True) 
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        #self.sigmoid = nn.Sigmoid()


    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the discriminator.
        It has the following architecture:
        Conv2d -> LeakyReLU -> Conv2d -> LeakyReLU -> Conv2d -> LeakyReLU -> Conv2d -> LeakyReLU -> Classifier 
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W] where B is batch size, C is number of classes,
                        H is height and W is width.
                        
        Returns:
            x (torch.Tensor): Output tensor of shape [B, 1, H', W'] where H' and W' are reduced dimensions after convolutions.
        """
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        #x = self.up_sample(x)
        #x = self.sigmoid(x) 

        return x