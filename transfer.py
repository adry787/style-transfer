#!/usr/bin/env python3
import torch, torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


class StyleNet(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=None).features
        self.s1 = vgg[:2]
        self.s2 = vgg[2:7]
        self.s3 = vgg[7:12]
        self.s4 = vgg[12:21]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        h1 = self.s1(x)
        h2 = self.s2(h1)
        h3 = self.s3(h2)
        h4 = self.s4(h3)
        return h1, h2, h3, h4


class Transfer:
    def __init__(self):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StyleNet().to(self.dev)
        self.tf = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    def gram(self, x):
        b, c, h, w = x.shape
        f = x.view(b * c, h * w)
        return torch.mm(f, f.t()) / (c * h * w)

    def run(self, cp, sp, steps=50):
        c = self.tf(Image.open(cp)).unsqueeze(0).to(self.dev)
        s = self.tf(Image.open(sp)).unsqueeze(0).to(self.dev)
        t = c.clone().requires_grad_(True)
        opt = torch.optim.Adam([t], lr=0.01)
        cf = self.model(c)
        sf = self.model(s)
        sg = [self.gram(f) for f in sf]
        for i in range(steps):
            tf = self.model(t)
            loss = torch.nn.functional.mse_loss(tf[2], cf[2]) + 1e6 * sum(
                torch.nn.functional.mse_loss(self.gram(f), g) for f, g in zip(tf, sg)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        T.ToPILImage()(t.squeeze().clamp(0, 1)).save("output.jpg")
        print("Saved output.jpg")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python transfer.py content.jpg style.jpg")
        exit()
    Transfer().run(sys.argv[1], sys.argv[2])
