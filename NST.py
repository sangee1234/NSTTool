import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.optim as optim
from VGG import CustomVGG
import torch

class NST:
    def __init__(self, style_path, content_path):
        #self.model = models.vgg19(pretrained=True).features
        self.style = style_path
        self.content = content_path
        preprocessor = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        self.style_image = preprocessor(Image.open(style_path).convert('RGB')).unsqueeze(0)
        self.content_image = preprocessor(Image.open(content_path).convert('RGB')).unsqueeze(0)
        self.new_image = self.content_image.clone().requires_grad_(True)
        self.model = CustomVGG()
        self.epoch = 100
        self.lr = 0.005
        self.content_weight = 0.2
        self.style_weight = 0.8
        self.optimizer = optim.Adam([self.new_image],lr=self.lr)

    def content_loss(self, new, original):
        return torch.mean((new-original)**2)

    def style_loss(self, new, original):
        batch_size, c, h, w = new.shape
        new_style = torch.mm(new.view(c ,h*w), original.view(c ,h*w).t())
        orig_style = torch.mm(original.view(c ,h*w), original.view(c ,h*w).t())
        return torch.mean((new_style-orig_style)**2)

    def total_loss(self, new_features, orig_content_features, orig_style_features):
        sl =0
        cl = 0
        for new,con,style in zip(new_features, orig_content_features, orig_style_features):
            cl += self.content_loss(new, con)
            sl += self.style_loss(new, style)

        return self.content_weight*cl + self.style_weight*sl


    def train(self):
        for i in range(self.epoch):
            new_features = self.model(self.new_image)
            content_features = self.model(self.content_image)
            style_features = self.model(self.style_image)
            loss = self.total_loss(new_features, content_features, style_features)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(i)
        return self.new_image
