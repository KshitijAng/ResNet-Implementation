import torch
import torch.nn as nn

# Define the Residual Block
# A bottleneck residual block is designed to reduce computation while maintaining the expressiveness of deep networks. 
# Instead of using two 3×3 convolutional layers like in ResNet-18 and ResNet-34, this block uses three convolutional layers.

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        # The identity_downsample is to handle situations where the input and output dimensions of a layer do not match. 
        super(block,self).__init__()
        self.expansion = 4 # The number of channels is expanded by 4 in the final layer.
        
        # First convolution: 1x1 conv (Compression Layer)
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels) # Batch Normalization for 2D Inputs
        
        # Second convolution: 3x3 conv (Feature Extraction)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1) # Down Sampling
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Third convolution: 1x1 conv (Expansion Layer)
        self.conv3 = nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=1,stride=1,padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.relu = nn.ReLU() # Activation Function
        self.identity_downsample = identity_downsample # # Used for downsampling if needed
    
    def forward(self, x):
        identity = x # Store original input for residual connection
        
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        # If input and output dimensions don't match, downsample the identity
        # Downsampling is needed in a residual block when the input and output dimensions do not match.            
        if self.identity_downsample is not None: 
            identity = self.identity_downsample(identity)
           
        x += identity # Add identity (skip connection) 
        x = self.relu(x)
        return x


# Stride refers to the number of pixels by which the convolutional filter moves across the input image or feature map.

# Stride = 1 → The filter moves one pixel at a time (default setting).
# Stride = 2 → The filter moves two pixels at a time, effectively reducing the spatial dimensions (downsampling).
# Stride > 2 → The filter moves even more pixels, further decreasing the output size.    


# Define the ResNet Model    
class ResNet(nn.Module):
    def __init__(self,block,layers,image_channels,num_classes):
        super(ResNet,self).__init__()
        self.in_channels = 64 # Initial number of channels
        
        self.conv1 = nn.Conv2d(image_channels,64,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # MaxPooling Layer to reduce feature map size
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        # Create ResNet layers
        self.layer1 = self.make_layer(block,layers[0],out_channels=64,stride=1)
        self.layer2 = self.make_layer(block,layers[1],out_channels=128,stride=2)
        self.layer3 = self.make_layer(block,layers[2],out_channels=256,stride=2)
        self.layer4 = self.make_layer(block,layers[3],out_channels=512,stride=2)
        
        # Average Pooling to reduce feature map to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # Fully Connected Layer (classifier)
        self.fc = nn.Linear(512*4,num_classes) # Final output shape
 
    def forward(self,x):
        x= self.conv1(x) # Initial Conv layer
        x =self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global Average Pooling
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)  # Flatten before FC layer
        x = self.fc(x) # Fully connected classifier
        
        return x
        
        
    def make_layer(self,block, num_residual_blocks, out_channels,stride):
        identity_downsample = None # We assume downsampling is not needed.
        layers = []
        
        # If the input and output dimensions are different, we need downsampling
        if stride != 1 or self.in_channels != out_channels * 4: 
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,out_channels*4,kernel_size=1,stride=stride),
                nn.BatchNorm2d(out_channels*4)
                )
        
        # First block of the layer (with downsampling if needed)    
        layers.append(block(self.in_channels,out_channels,identity_downsample,stride))
        self.in_channels = out_channels * 4 # Update input channels for next block
        
        # Remaining blocks (no downsampling)
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels,out_channels)) #256 -> 64, 64*4 (256) again
            
        return nn.Sequential(*layers)  # Stack all blocks
            
def ResNet50(img_channels=3,num_classes=1000):
    return ResNet(block,[3,4,6,3], img_channels, num_classes)

# def ResNet101(img_channels=3,num_classes=1000):
#     return ResNet(block,[3,4,23,3], img_channels, num_classes)

def test():
    net = ResNet50()
    x = torch.randn(2,3,224,224)  # Create a batch of 2 images of size 224x224
    y = net(x).to('mps')  # Forward pass 
    print(y.shape) # Output shape should be (2, 1000)
    
test()