import torch
import torchvision
import torch.nn.functional as F

import numpy as np
from scipy.stats import truncnorm

def reconstruct(model, out_path, num, add_small_noise=False):
    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device
        dataset_size = model.embeddings.weight.size()[0]

        indices = torch.arange(num)

        assert type(indices) == torch.Tensor

        indices = indices.to(device)        
        embeddings = model.embeddings(indices)
        batch_size = embeddings.size()[0]

        if add_small_noise:
            embeddings += torch.randn(embeddings.size(), device=device) * 0.01

        image_tensors = model(embeddings)
        # torchvision.utils.save_image(
        #     image_tensors,
        #     f"{out_path}reconstruct_{i}.jpg",
        #     nrow=1,#int(batch_size ** 0.5),
        #     normalize=True,
        # )

        for i, val in enumerate(image_tensors):
            torchvision.utils.save_image(
                val,
                f"{out_path}reconstruct_{i}.jpg",
                nrow=1,
                normalize=True,
            )
        
#see https://github.com/nogu-atsu/SmallGAN/blob/2293700dce1e2cd97e25148543532814659516bd/gen_models/ada_generator.py#L37-L53
def interpolate(model, out_path, source, dist, trncate=0.4, num=5):
    with torch.no_grad():
            model.eval()
            device = next(model.parameters()).device
            dataset_size = model.embeddings.weight.size()[0]
            indices = torch.tensor([source, dist], device=device)

            indices = indices.to(device) 
            embeddings = model.embeddings(indices)
            embeddings = embeddings[[0]] * torch.linspace(1, 0, num, device=device)[:, None] + embeddings[[1]] * torch.linspace(0, 1, num, device=device)[:, None]
            batch_size = embeddings.size()[0]
            image_tensors = model(embeddings)

            for i, val in enumerate(image_tensors):
                torchvision.utils.save_image(
                    val,
                    f"{out_path}interpolate_{i}.jpg",
                    nrow=1,
                    normalize=True,
                )

#from https://github.com/nogu-atsu/SmallGAN/blob/2293700dce1e2cd97e25148543532814659516bd/gen_models/ada_generator.py#L37-L53        
def random(model, out_path, tmp=0.4, num=9, prefix=1, truncate=False):
    with torch.no_grad():
        model.eval()
        device = next(model.parameters()).device
        dim_z = model.embeddings.weight.size(1)
        if truncate:
            embeddings = truncnorm(-tmp, tmp).rvs(num * dim_z).astype("float32").reshape(num, dim_z)
        else:
            embeddings = np.random.normal(0, tmp, size=(num, dim_z)).astype("float32")
        embeddings = torch.tensor(embeddings,device=device)

        image_tensors = model(embeddings)
        # torchvision.utils.save_image(
        #         image_tensors,
        #         f"{out_path}random.jpg",
        #         nrow=1,
        #         normalize=True,
        #     )

        _preds = []
        for i, val in enumerate(image_tensors):
            _preds.append(get_predictions(val))
            
            
            # torchvision.utils.save_image(
            #     val,
            #     f"{out_path}random_{prefix}_{i}.jpg",
            #     nrow=1,
            #     normalize=True,
            # )
        
        preds = torch.cat(_preds).numpy()

        print(preds)

        probs = preds.max(axis=1)
        indices = probs.argsort(axis=0)

        print(indices)

        new_samples = []
        for item in indices:
            new_samples.append(image_tensors[item])

        # image_tensors = new_samples[: new_samples // 2]
        # for i, val in enumerate(image_tensors):
        #     torchvision.utils.save_image(
        #         val,
        #         f"{out_path}random_{prefix}_{i}.jpg",
        #         nrow=1,
        #         normalize=True,
        #     )

def get_predictions(outputs):
    dist1 = F.softmax(outputs, dim=1)
    preds = dist1.detach().cpu()

    return preds