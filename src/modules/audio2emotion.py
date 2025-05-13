import os
import sys
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.abspath("../")))

class Audio2EmotionModel(nn.Module):
    def __init__(self, num_classifier_layers=5, num_classifier_channels=2048, num_emotion_classes=8):
        super().__init__()
        self.num_emotion_classes = num_emotion_classes

        if num_classifier_layers == 1:
            self.layers = nn.Linear(1024, self.num_emotion_classes)
        else:
            layer_list = [
                nn.Linear(1024, num_classifier_channels),
                nn.ReLU()
            ]
            for _ in range(num_classifier_layers - 2):
                layer_list.append(nn.Linear(num_classifier_channels, num_classifier_channels))
                layer_list.append(nn.ReLU())
            layer_list.append(nn.Linear(num_classifier_channels, self.num_emotion_classes))
            self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.layers(x)
        return x
