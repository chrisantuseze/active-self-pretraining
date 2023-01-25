import torch
import torchvision
import numpy as np
from scipy.stats import truncnorm

def reconstruct(model, out_path, num, add_small_noise=False):
    for i in range(1, num+1):
        with torch.no_grad():
            model.eval()
            device = next(model.parameters()).device
            dataset_size = model.embeddings.weight.size()[0]

            i = torch.arange(i)

            assert type(i) == torch.Tensor

            i = i.to(device)        
            embeddings = model.embeddings(i)
            batch_size = embeddings.size()[0]

            if add_small_noise:
                embeddings += torch.randn(embeddings.size(), device=device) * 0.01

            image_tensors = model(embeddings)

            print(len(image_tensors), image_tensors.shape)
            torchvision.utils.save_image(
                image_tensors,
                f"_{i}_" + out_path,
                nrow=1,#int(batch_size ** 0.5),
                normalize=True,
            )
        
#see https://github.com/nogu-atsu/SmallGAN/blob/2293700dce1e2cd97e25148543532814659516bd/gen_models/ada_generator.py#L37-L53
def interpolate(model, out_path, source, dist, trncate=0.4, num=5):
    for i in range(1, num+1):
        with torch.no_grad():
            model.eval()
            device = next(model.parameters()).device
            dataset_size = model.embeddings.weight.size()[0]
            indices = torch.tensor([source,dist],device=device)
            indices = indices.to(device) 
            embeddings = model.embeddings(indices)
            embeddings = embeddings[[0]] * torch.linspace(1, 0, i, device=device)[:, None] + embeddings[[1]]* torch.linspace(0, 1, i, device=device)[:, None]
            batch_size = embeddings.size()[0]
            image_tensors = model(embeddings)
            torchvision.utils.save_image(
                image_tensors,
                f"_{i}_" + out_path,
                nrow=1,#batch_size,
                normalize=True,
            )

#from https://github.com/nogu-atsu/SmallGAN/blob/2293700dce1e2cd97e25148543532814659516bd/gen_models/ada_generator.py#L37-L53        
def random(model, out_path, tmp=0.4, num=9, truncate=False):
    for i in range(1, num+1):
        with torch.no_grad():
            model.eval()
            device = next(model.parameters()).device
            dataset_size = model.embeddings.weight.size()[0]
            dim_z = model.embeddings.weight.size(1)
            if truncate:
                embeddings = truncnorm(-tmp, tmp).rvs(i * dim_z).astype("float32").reshape(i, dim_z)
            else:
                embeddings = np.random.normal(0, tmp, size=(i, dim_z)).astype("float32")
            embeddings = torch.tensor(embeddings,device=device)
            batch_size = embeddings.size()[0]
            image_tensors = model(embeddings)
            torchvision.utils.save_image(
                    image_tensors,
                    f"_{i}_" + out_path,
                    nrow=1,#int(batch_size ** 0.5),
                    normalize=True,
                )