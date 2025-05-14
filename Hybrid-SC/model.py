import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ResNet50DualBranch(nn.Module):
    def __init__(self, num_classes=1000, proj_dim=128, hidden_dim=2048):
        super().__init__()

        # Backbone ResNet-50 sans couche FC
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove fc

        feat_dim = 2048  # Résultat du ResNet-50

        # Classification branch
        self.classifier = nn.Linear(feat_dim, num_classes)

        # Projection branch (SIMCLR style)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

        # Learnable parameters: parameters for the prototypes
        # (num_classes, proj_dim) for each class
        self.learnable_vectors = nn.Parameter(torch.randn(num_classes, proj_dim))
        # Optionnel : normaliser les vecteurs au départ
        nn.init.kaiming_normal_(self.learnable_vectors, mode='fan_out', nonlinearity='relu')

    def forward(self, x_cls=None, x_proj=None):
        out_cls, out_proj, proto = None, None, None

        # Branche classification
        if x_cls is not None:
            feat_cls = self.backbone(x_cls)
            feat_cls = feat_cls.view(feat_cls.size(0), -1)
            out_cls = self.classifier(feat_cls)

        # Branche projection
        if x_proj is not None:
            feat_proj = self.backbone(x_proj)
            feat_proj = feat_proj.view(feat_proj.size(0), -1)
            proj = self.projector(feat_proj)
            out_proj = F.normalize(proj, dim=1)
            proto = F.normalize(self.learnable_vectors, dim=1)  # L2 norm optional

        return out_cls, out_proj, proto


class CombinedPSCLoss(nn.Module):
    def __init__(self, alpha=1.0, temperature=0.1):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits, z, labels, prototypes):
        """
        logits     : [B, num_classes] — sortie de la branche classification
        z          : [B, proj_dim]    — sortie normalisée de la branche projection
        labels     : [B]              — labels ground truth
        prototypes : [num_classes, proj_dim] — prototypes normalisés
        """
        # Cross entropy sur les logits
        ce_loss = self.cross_entropy(logits, labels)

        # PSC loss (déjà normalisé)
        logits_proj = torch.matmul(z, prototypes.T)  # [B, C]
        logits_proj /= self.temperature

        # Masque pour exclure la classe cible du dénominateur
        B, C = logits_proj.size()
        mask = torch.ones_like(logits_proj).bool()
        mask[torch.arange(B), labels] = False

        numerator = torch.exp(logits_proj[torch.arange(B), labels])  # [B]
        denominator = torch.exp(logits_proj[mask].view(B, C - 1)).sum(dim=1)  # [B]
        psc_loss = -torch.log(numerator / denominator).mean()

        # Combine
        total_loss = (1-self.alpha)*ce_loss + self.alpha * psc_loss
        return total_loss