# (CASL): A Data-Efficient Domain and Task Adaptation Framework Using Active Learning and Self-Supervised Learning
This is the repository for the research on adapting pretrained models to different domains and tasks whilst ensuring the data budget required is reduced by a considerable amount. In this work, a self-supervised learner learns representations using unannotated data samples. A new learner is used to finetune the weights learned by the previous learner on a different domain and task. This is done whilst ensuring the second learner is trained using few samples through active learning.

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