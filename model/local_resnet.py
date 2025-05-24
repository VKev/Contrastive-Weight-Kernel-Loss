import torch
import torch.nn as  nn
import torch.nn.functional as F

class BottleneckLocal(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(BottleneckLocal, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class BlockLocal(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(BlockLocal, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x

class VectorPad(nn.Module):
    def __init__(self, target_dim: int):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, x):
        f = x.size(1)
        if f < self.target_dim:              
            pad = (0, self.target_dim - f)
            x = F.pad(x, pad, value=0.)
        elif f > self.target_dim:
            x = x[:, :self.target_dim]
        return x

class ResNetLocal(nn.Module):
    def __init__(self, ResBlockLocal, layer_list, num_classes, num_channels=3):
        super(ResNetLocal, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlockLocal, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlockLocal, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlockLocal, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlockLocal, layer_list[3], planes=512, stride=2)
        
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            VectorPad(512 * ResBlockLocal.expansion),
            nn.Linear(512 * ResBlockLocal.expansion, num_classes)
        )
        
    def forward_conv1(self, x):
        x = self.conv1(x)
        return x

    def forward_layer1(self, x):
        return self.layer1(x)

    def forward_layer2(self, x):
        return self.layer2(x)

    def forward_layer3(self, x):
        return self.layer3(x)

    def forward_layer4(self, x):
        return self.layer4(x)

    def forward_pool_fc(self, x):
        x = self.final(x)
        return x
    
    def forward(self, x):
        x = self.conv1(x)
        # x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final(x)
        
        return x
        
    def _make_layer(self, ResBlockLocal, BlockLocals, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlockLocal.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlockLocal.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlockLocal.expansion)
            )
            
        layers.append(ResBlockLocal(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlockLocal.expansion
        
        for i in range(BlockLocals-1):
            layers.append(ResBlockLocal(self.in_channels, planes))
            
        return nn.Sequential(*layers)

        
        
def ResNetLocal50(num_classes, channels=3):
    return ResNetLocal(BottleneckLocal, [3,4,6,3], num_classes, channels)
    
def ResNetLocal101(num_classes, channels=3):
    return ResNetLocal(BottleneckLocal, [3,4,23,3], num_classes, channels)

def ResNetLocal152(num_classes, channels=3):
    return ResNetLocal(BottleneckLocal, [3,8,36,3], num_classes, channels)

if __name__ == "__main__":
    batch_size = 8
    num_classes = 100
    input_tensor = torch.randn(batch_size, 3, 32, 32)  # dummy input
    
    # Instantiate model
    model = ResNetLocal50(num_classes=num_classes, channels=3)
    
    # Forward pass
    # outputs = model(input_tensor)
    
    output1 = model.forward_conv1(input_tensor)
    output1 = model.forward_pool_fc(output1)
    
    print("Output shape:", output1.shape)  # should be [batch_size, num_classes]