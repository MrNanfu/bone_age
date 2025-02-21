import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.stats as stats
from tqdm import tqdm

class SIMBA(nn.Module):

    def __init__(self, num_classes=1, aux_logits=False, transform_input=False, chronological_age=True, gender_multiplier=True, use_gut_microbiome=True, use_pe_performance=True):
        super(SIMBA, self).__init__()
        # Inception
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.chronological_age = chronological_age
        self.gender_multiplier = gender_multiplier

        self.Conv2d_1a_3x3 = nn.ModuleList()
        self.Conv2d_2a_3x3 = nn.ModuleList()
        self.Conv2d_2b_3x3 = nn.ModuleList()
        self.Conv2d_3b_1x1 = nn.ModuleList()
        self.Conv2d_4a_3x3 = nn.ModuleList()

        for x in range(2):
            self.Conv2d_1a_3x3.append(BasicConv2d(1, 32, kernel_size=3,
                                      stride=2, padding=0))
            self.Conv2d_2a_3x3.append(BasicConv2d(32, 32,
                                      kernel_size=3))
            self.Conv2d_2b_3x3.append(BasicConv2d(32, 64, kernel_size=3,
                                      padding=1))
            self.Conv2d_3b_1x1.append(BasicConv2d(64, 80,
                                      kernel_size=1))
            self.Conv2d_4a_3x3.append(BasicConv2d(80, 192,
                                      kernel_size=3))

        self.Conv2d_5a_1x1 = BasicConv2d(192*2, 192, kernel_size=1)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)

        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        fc_1_size = 100352
        
        # Gender
        if gender_multiplier:
            self.gender = Multiplier(1)
            fc_1_size += 1
        else:
            self.gender = DenseLayer(1, 32)
            fc_1_size += 32

        # Chronological Age
        if chronological_age :
            self.chronological = Multiplier(1)
            fc_1_size += 1
        
        # 肠道菌群 extractor 层
        self.use_gut_microbiome = use_gut_microbiome
        if self.use_gut_microbiome:
            # self.gut_extractor = nn.Linear(48, 64)  # 处理 PUA + shannon
            # fc_1_size += 64
            self.gut_extractor = GutMicrobiomeModule(input_dim=48, output_dim=128)
            fc_1_size += 128
            
        # 运动表现 extractor 层
        self.use_pe_performance = use_pe_performance
        if self.use_pe_performance:
            # self.pe_extractor = nn.Linear(6, 12)  # 处理 PUA + shannon
            # fc_1_size += 12
            self.pe_extractor = PhysicalPerformanceModule(input_dim=6, output_dim=32)
            fc_1_size += 32

        self.fc_1 = DenseLayer(fc_1_size, 1000)
        self.fc_2 = DenseLayer(1000, 1000)
        self.fc_3 = nn.Linear(1000, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.numel()))
                values = values.view(m.weight.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y, z, gut, pe):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229/0.5)+(0.485-0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224/0.5)+(0.456-0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225/0.5)+(0.406-0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        first_block = []
        x = torch.split(x, 1, dim=1)
        x = list(x)
        for index in range(2):
            into = x[index]
            into = self.Conv2d_1a_3x3[index](into)
            into = self.Conv2d_2a_3x3[index](into)
            into = self.Conv2d_2b_3x3[index](into)
            into = F.max_pool2d(into, kernel_size=3, stride=2)
            into = self.Conv2d_3b_1x1[index](into)
            into = self.Conv2d_4a_3x3[index](into)
            into = F.max_pool2d(into, kernel_size=3, stride=2)
            first_block.append(into)
            #x = into
        
        x = torch.cat(first_block, dim=1)
        x = self.Conv2d_5a_1x1(x)

        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=2)
        x = x.view(x.size(0), -1)

        y = self.gender(y)
        
        features = [x, y]  # 图像特征 & 性别

        if self.chronological_age:
            z = self.chronological(z)
            features.append(z)
        if self.use_gut_microbiome:
            gut = self.gut_extractor(gut)
            features.append(gut)
        if self.use_pe_performance:
            pe = self.pe_extractor(pe)
            features.append(pe)

        x = torch.cat(features, dim=1)
        
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        if self.training and self.aux_logits:
            return x, aux
        return x
        

class Multiplier(nn.Module):

    def __init__(self, size):
        super(Multiplier, self).__init__()
        self.multiplier = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x):
        x = x * self.multiplier
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features,
                                       kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7),
                                       padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1),
                                       padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1),
                                          padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7),
                                          padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1),
                                          padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7),
                                          padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7),
                                         padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1),
                                         padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3),
                                        padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1),
                                        padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3),
                                           padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1),
                                           padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class DenseLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.linear(x)
        return F.relu(x, inplace=True)
    
    
class GutMicrobiomeModule(nn.Module):
    def __init__(self, input_dim=48, output_dim=128, hidden_dim=64):
        super(GutMicrobiomeModule, self).__init__()
        
        # 多层 MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(0.3)  # 预防过拟合
        self.relu = nn.ReLU()

        # 自注意力层
        self.attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=4, batch_first=True)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)  # (batch, output_dim)
        
        # 添加注意力机制
        x = x.unsqueeze(1)  # (batch, 1, output_dim)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(1)  # (batch, output_dim)
        
        return x
    
    
class PhysicalPerformanceModule(nn.Module):
    def __init__(self, input_dim=6, output_dim=32, hidden_dim=16):
        super(PhysicalPerformanceModule, self).__init__()

        # 1D CNN 提取局部特征
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(16 * input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

        # 注意力层
        self.attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=2, batch_first=True)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)  # (batch, output_dim)

        # 添加注意力机制
        x = x.unsqueeze(1)  # (batch, 1, output_dim)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(1)  # (batch, output_dim)

        return x

