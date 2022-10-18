import torchvision


def resnet_backbone(name, pretrained=False):
    weights18 = torchvision.models.resnet.ResNet18_Weights.DEFAULT if pretrained else None
    weights50 = torchvision.models.resnet.ResNet50_Weights.DEFAULT if pretrained else None

    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),#weights=weights18),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),#weights=weights50),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]
