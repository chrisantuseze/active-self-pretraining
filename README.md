# (CASL): A Data-Efficient Domain and Task Adaption Framework Using Active Learning and Self-Supervised Learning
This is the repository for the research on reducing the data budget required on pretraining a second layer pretrained model. Here I used active learning to reduce the data budget required in pretraining a self-supervised model

 # Continual Active Self-Learning (CASL): An Active and Self-Supervised Learning Approach for Adapting Models to Domains and Tasks

To install packages, for now do 'conda install <package>' on the individual packages

## Usage
Run the following command to setup a conda environment:
```
sh setup.sh
conda activate casl
```

Or alternatively with pip:
```
pip install -r requirements.txt
```

Then, simply run for single GPU or CPU training:
```
python main.py
```