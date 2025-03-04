import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SIMBA(nn.Module):
    def __init__(self, num_classes=1, feature_extractor='resnet', aux_logits=False, transform_input=False, chronological_age=False, gender_multiplier=False, use_gender=False, use_gut_microbiome=False, use_pe_performance=False, use_correlation=False, use_image=True):
        super(SIMBA, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.chronological_age = chronological_age
        self.use_gender = use_gender
        self.gender_multiplier = gender_multiplier
        self.use_gut_microbiome = use_gut_microbiome
        self.use_pe_performance = use_pe_performance
        self.use_correlation = use_correlation
        self.use_image = use_image  # 控制是否使用图像
        self.feature_extractor_name = feature_extractor.lower()
        
        if self.use_image:
            self.feature_extractor = self._initialize_feature_extractor()
            feature_dim = self._get_feature_dim()
        else:
            feature_dim = 0  # 不使用图像时初始大小为0
        if self.use_gender:
            # Gender
            if gender_multiplier:
                self.gender = Multiplier(1)
                feature_dim += 1
            else:
                self.gender = DenseLayer(1, 32)
                feature_dim += 32

        # Chronological Age
        if chronological_age:
            self.chronological = Multiplier(1)
            feature_dim += 1
        
        # 肠道菌群 extractor 层
        if self.use_gut_microbiome:
            self.gut_extractor = GutMicrobiomeModule(input_dim=46, output_dim=128)
            feature_dim += 128
            
        # 运动表现 extractor 层
        if self.use_pe_performance:
            self.pe_extractor = PhysicalPerformanceModule(input_dim=6, output_dim=32)
            feature_dim += 32
        
        # 相关特征 extractor 层
        if self.use_correlation:
            self.correlation_extractor = CorrelationFeatureModule(input_dim=19, output_dim=64)
            feature_dim += 64
        
        self.fc_1 = DenseLayer(feature_dim, 1000)
        self.fc_2 = DenseLayer(1000, 1000)
        self.fc_3 = nn.Linear(1000, num_classes)
    
    def _initialize_feature_extractor(self):
        if self.feature_extractor_name == 'inception':
            model = models.inception_v3(pretrained=True, aux_logits=True)
            model.fc = nn.Identity()
        elif self.feature_extractor_name == 'resnet':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Identity()
        elif self.feature_extractor_name == 'vgg':
            model = models.vgg16(pretrained=True)
            model.classifier = nn.Identity()
        elif self.feature_extractor_name == 'efficientnet':
            model = models.efficientnet_b0(pretrained=True)
            model.classifier = nn.Identity()
        elif self.feature_extractor_name == 'vit':
            model = models.vit_b_16(pretrained=True)
            model.heads = nn.Identity()
        else:
            raise ValueError(f"Unsupported feature extractor: {self.feature_extractor_name}")
        
        return model
    
    def _get_feature_dim(self):
        if self.feature_extractor_name in ['inception', 'resnet']:
            return 2048
        elif self.feature_extractor_name == 'vgg':
            return 25088
        elif self.feature_extractor_name == 'efficientnet':
            return 1280
        elif self.feature_extractor_name == 'vit':
            return 768
        return 0
    
    def forward(self, x=None, y=None, z=None, gut=None, pe=None, cor=None):
        features = []
        if self.use_image:
            if x.shape[1] == 2:  # 检测输入通道数
                x = torch.cat([x, torch.zeros_like(x[:, :1, :, :])], dim=1)  # 添加全零通道
            x = self.feature_extractor(x)
            if isinstance(x, tuple):  # 如果是 (output, aux_output)
                x = x[0]  # 取主输出
            x = x.view(x.size(0), -1)

            features.append(x)
        if self.use_gender:
            y = self.gender(y)
            features.append(y)
        
        if self.chronological_age:
            z = self.chronological(z)
            features.append(z)
        if self.use_gut_microbiome:
            gut = self.gut_extractor(gut)
            features.append(gut)
        if self.use_pe_performance:
            pe = self.pe_extractor(pe)
            features.append(pe)
        if self.use_correlation:
            cor = self.correlation_extractor(cor)
            features.append(cor)

        x = torch.cat(features, dim=1)
        
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)
        return x
    
    
class Multiplier(nn.Module):

    def __init__(self, size):
        super(Multiplier, self).__init__()
        self.multiplier = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x):
        x = x * self.multiplier
        return x
    
    
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
    
    
class CorrelationFeatureModule(nn.Module):
    def __init__(self, input_dim=20, output_dim=64, hidden_dim=32):
        super(CorrelationFeatureModule, self).__init__()

        # MLP 提取特征
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        
        self.dropout = nn.Dropout(0.3)
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

        # 添加自注意力机制
        x = x.unsqueeze(1)  # (batch, 1, output_dim)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(1)  # (batch, output_dim)

        return x

class DenseLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.linear(x)
        return F.relu(x, inplace=True)