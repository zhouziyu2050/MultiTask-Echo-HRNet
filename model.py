import timm
import torch
from torch import nn
from torch.nn import functional as functional


class HRNetW18Lite(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("hrnet_w18", pretrained=False, features_only=True)
        channels = self.backbone.feature_info.channels()[1:]
        self.scale_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, 32, kernel_size=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            ) for channel in channels
        ])
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.heads = nn.ModuleList([nn.Conv2d(128, 2, kernel_size=1) for task_index in range(6)])

    def forward(self, images, task_ids):
        features = self.backbone(images)[1:]
        grid = features[0].shape[-2:]
        aligned = [features[0]] + [
            functional.interpolate(feature, size=grid, mode="bilinear", align_corners=False)
            for feature in features[1:]
        ]
        fused = torch.cat([
            projection(feature) for feature, projection in zip(aligned, self.scale_projs)
        ], dim=1)
        decoded = self.decoder(fused)
        output = decoded.new_zeros((images.shape[0], 2, *grid))
        for task_id in task_ids.unique():
            selected = task_ids == task_id
            output[selected] = self.heads[int(task_id.item())](decoded[selected])
        return functional.interpolate(output, size=images.shape[-2:], mode="bilinear", align_corners=False)


