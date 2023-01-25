import torch.nn as nn
import torchvision.models as models

class CustomVGG(nn.Module):
    #To get the intermediate layers output, so that it can be used to extract style n content
    def __init__(self):
        super().__init__()
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features_required = []
        for n, feature in enumerate(self.model):
            x = feature(x)
            if n in [0,5,10,19,28]:
                features_required.append(x)
        return features_required
