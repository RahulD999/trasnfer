import os

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets
import torchvision.transforms as T
from .helpers import get_data_location


class Predictor(nn.Module):
    def __init__(self, model, class_names, mean, std):
        super().__init__()
        self.model = model.eval()
        self.class_names = class_names
        self.transforms = nn.Sequential(
            T.Resize([256, ]),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean.tolist(), std.tolist())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Ensure x is the right shape
            if x.dim() == 3:
                x = x.unsqueeze(0)
            elif x.dim() == 5:
                x = x.squeeze(1)
            
            x = self.transforms(x)
            x = self.model(x)
            x = F.softmax(x, dim=1)
            return x


def predictor_test(test_dataloader, model_reloaded):
    import numpy as np
    from tqdm import tqdm
    import torch
    
    pred = []
    truth = []
    with torch.no_grad():
        for x, y in tqdm(test_dataloader, total=len(test_dataloader), leave=True, ncols=80):
            # Ensure x is the right shape
            if x.dim() == 5:
                x = x.squeeze(1)
            
            # Move data to the same device as the model
            x = x.to(next(model_reloaded.parameters()).device)
            
            softmax = model_reloaded(x)
            
            idx = softmax.argmax(dim=1)
            
            pred.extend(idx.cpu().numpy())
            truth.extend(y.cpu().numpy())

    pred = np.array(pred)
    truth = np.array(truth)

    print(f"Accuracy: {(pred==truth).sum() / pred.shape[0]}")

    return pred, truth

######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    from .model import MyModel
    from .helpers import compute_mean_and_std

    mean, std = compute_mean_and_std()

    model = MyModel(num_classes=3, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)
    

    predictor = Predictor(model, class_names=['a', 'b', 'c'], mean=mean, std=std)

    out = predictor(images)
    
    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 3]
    ), f"Expected an output tensor of size (2, 3), got {out.shape}"

    assert torch.isclose(
        out[0].sum(),
        torch.Tensor([1]).squeeze()
    ), "The output of the .forward method should be a softmax vector with sum = 1"
