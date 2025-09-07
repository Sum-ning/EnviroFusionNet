import torch
import torch.nn as nn
from models.image_encoder import ImageEncoder
from models.sequence_encoder import TransformerClassifier
from models.environment_encoder import EnvironmentEncoder
from models.caff_module import CAFF
from models.cmac_module import CMAC

class EnviroFusionNet(nn.Module):
    def __init__(self, num_classes=10):
        super(EnviroFusionNet, self).__init__()
        self.image_encoder = ImageEncoder()
        self.sequence_encoder = TransformerClassifier(input_dim=72, seq_len=260, num_classes=num_classes)
        self.environment_encoder = EnvironmentEncoder()
        self.caff = CAFF(feature_dim=64)
        self.cmac = CMAC(dim=64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, image, sequence, env):
        I_g = self.image_encoder(image)        # [B, L1, D]
        S_g = self.sequence_encoder(sequence)  # [B, L2, D]
        E_g = self.environment_encoder(env)    # [B, D]

        F_g = self.caff(I_g, S_g)              # [B, D]
        Y_g = self.cmac(F_g, E_g)              # [B, D]
        out = self.classifier(Y_g)             # [B, num_classes]
        return out
