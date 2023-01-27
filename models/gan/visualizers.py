import torch
import torchvision
import numpy as np
from scipy.stats import truncnorm

def reconstruct(model, out_path, num, add_small_noise=False):
    for i in range(num):
        with torch.no_grad():
            model.eval()
            device = next(model.parameters()).device
            dataset_size = model.embeddings.weight.size()[0]

            indices = torch.arange(1)

            assert type(indices) == torch.Tensor

            indices = indices.to(device)        
            embeddings = model.embeddings(indices)
            batch_size = embeddings.size()[0]

            if add_small_noise:
                embeddings += torch.randn(embeddings.size(), device=device) * 0.01

            image_tensors = model(embeddings)

            print(len(image_tensors), image_tensors)
            torchvision.utils.save_image(
                image_tensors,
                f"{out_path}reconstruct_{i}.jpg",
                nrow=1,#int(batch_size ** 0.5),
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
def random(model, out_path, tmp=0.4, num=9, truncate=False):
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
        for i, val in enumerate(image_tensors):
            torchvision.utils.save_image(
                val,
                f"{out_path}random_{i}.jpg",
                nrow=1,
                normalize=True,
            )