import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm


def collect_feature(data_loader: DataLoader, classifier,feature_extractor: nn.Module,
                    device: torch.device, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return

    Returns:
        Features in shape (min(len(data_loader), max_num_features * mini-batch size), :math:`|\mathcal{F}|`).
    """
    classifier.eval()
    feature_extractor.eval()
    all_features = []
    all_labels = []
    pres=[]
    with torch.no_grad():
        # for i, (images, target) in enumerate(tqdm.tqdm(data_loader)):
        for i, (images, target) in enumerate(data_loader):
            if max_num_features is not None and i >= max_num_features:
                break
            if(type(images)==list):
                images = images[0].to(device)
            else:
                images = images.to(device)
            feature = feature_extractor(images)
            pre=classifier(images)
            all_features.append(feature)
            all_labels.append(target)
            pres.append(pre)
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0),torch.cat(pres, dim=0)
