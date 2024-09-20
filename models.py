import torch.nn as nn
import torch.nn.functional as F


criterion_classification = nn.CrossEntropyLoss()
criterion_regression = nn.SmoothL1Loss()


def compute_loss(class_pred, class_true, bbox_pred, bbox_true):
    classification_loss = criterion_classification(class_pred, class_true)

    iou = calculate_iou(boox_pred, bbox_true)
    bbox_loss = 1 - iou.mean()

    total_loss = classification_loss + 0.5 * bbox_loss
    return total_loss


def calculate_iou(bbox_pred, bbox_true):
    # Intersection coordinates
    x1 = torch.max(boox_pred[:, 0], bbox_true[:, 0])
    y1 = torch.max(bbox_pred[:, 1], bbox_true[:, 1])
    x2 = torch.max(bbox_pred[:, 2], bbox_true[:, 2])
    y2 = torch.max(bbox_pred[:, 3], bbox_true[:, 3])

    # intedrsection area
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # areas of bbox
    bbox_pred_area = (bbox_pred[:, 2] - bbox_pred[:, 0]) * (bbox_pred[:, 3] - bbox_pred[:, 1])
    bbox_true_area = (bbox_true[:, 2] - bbox_true[:, 0]) * (bbox_true[:, 3] - bbox_true[:, 1])

    union = bbox_pred_area + bbox_true_area - intersection

    iou = intersection / union.clamp(min=1e-6)  # no devision by 0

    return iou


class VGG16(nn.Module):
    def __init__(self, device, num_class):
        super(VGG16, self).__init__()
        self.device = device
        self.num_class = num_class

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, self.num_class)
        )

        self.bbox_regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)

        x = torch.flatten(x, 1)

        class_out = self.classifier(x)
        bbox_out = self.bbox_regressor(x)

        return class_out, bbox_out
