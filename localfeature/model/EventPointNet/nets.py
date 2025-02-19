###
#
#       @Brief          nets.py
#       @Details        EventPointNet network 
#       @Org            Robot Learning Lab(https://rllab.snu.ac.kr), Seoul National University
#       @Author         Howoong Jun (howoong.jun@rllab.snu.ac.kr)
#       @Date           Mar. 18, 2021
#       @Version        v0.11
#
###

import torch

class CEventPointNet(torch.nn.Module):
    def __init__(self):
        super(CEventPointNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1_1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # self.conv2_2_1 = torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        # self.conv2_2_2 = torch.nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        
        self.conv3_1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # self.conv3_2_1 = torch.nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        # self.conv3_2_2 = torch.nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)

        self.conv4_1 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        # self.convDsc1 = torch.nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)
        # self.convDsc2 = torch.nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2)

        self.convKp1 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convKp2 = torch.nn.Conv2d(256, 65, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        kpt = self.pool(x)
        kpt = self.relu(self.conv2_1(kpt))
        kpt = self.relu(self.conv2_2(kpt))
        kpt = self.pool(kpt)
        kpt = self.relu(self.conv3_1(kpt))
        kpt = self.relu(self.conv3_2(kpt))
        kpt = self.pool(kpt)
        kpt = self.relu(self.conv4_1(kpt))
        kpt = self.relu(self.conv4_2(kpt))

        # desc = self.relu(self.conv2_2_1(x))
        # desc = self.relu(self.conv2_2_2(desc))
        # desc = self.relu(self.conv3_2_1(desc))
        # desc = self.relu(self.conv3_2_2(desc))

        kpt = self.relu(self.convKp1(kpt))
        kpt = self.convKp2(kpt)

        # desc = self.relu(self.convDsc1(desc))
        # desc = self.convDsc2(desc)

        # descNorm = torch.norm(desc, p=2, dim=1)
        # desc = desc.div(torch.unsqueeze(descNorm, 1))

        return kpt#, desc

        