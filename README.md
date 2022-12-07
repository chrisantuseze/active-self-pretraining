# Bootstrap Your Own Learning: An Active Learning Framework for Data-Efficient Learning Using Self-Supervision
This is the code repository for our work where we proposed a framework that enables the pretraining of a model in a self-supervised fashion and subsequently using the learned weights to select samples to be used in kickstarting an active learning process in a low data budget regime whilst ensuring a good performance is achieved in the downstream task.

## Usage
Run the following command to install the dependencies for the project:

```
pip install -r requirements.txt
```

Then, simply run for single GPU or CPU training:
```
python main.py
```

## Datasets

We used two datasets; ImageNet and Cifar10.

The datasets are available from this link: https://drive.google.com/drive/folders/1kIlQXoOl7tl8GwX-PKvvZR-Aklm16F59?usp=share_link

### Steps to get and use the dataset
1. Create a folder named "dataset" in the project root folder
2. Download and extract the dataset(s) and then move them to the newly created directory.
3. Ensure you rename the ImageNet folder to "imagenet" and the Cifar10 folder to "cifar10v2"

You also need to create a folder named "save" in the project root folder where the checkpoints and other code generated files would be saved in.

