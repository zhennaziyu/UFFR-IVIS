# Unsupervised Few-shot Food Recognition with Intra-Class Variationand Inter-Class Similarity Modeling (UFFR-IVIS)
The code repository for "UFFR-IVIS" 
## Overview
UFFR-IVIS consists of  (1) dual diversity-injected support/query representation learning that introduces instance-level and representation-level diversities for the representation learning of support/query instance to model the characteristics of high intra-class variation; and  (2) dual regularization-enhanced meta learning that designs two regularizations: auxiliary task-based intra-class regularization and similarity-guided inter-class regularization to regularize the intra-class variation and inter-class similarity modeling, respectively.

## Dataset

### MiniFood and LargeFood
Downlado data here [https://drive.google.com/drive/folders/1Bz7cygevBJd9_gEKhTbejQqW45m9FEIH?usp=drive_link]

## Training scripts

python train.py --eval_all --unsupervised --batch_size 64 --augment 'ws' - --max_epoch 300 --model_class ProtoNet --backbone_class ConvNet --dataset Food2K --way 5 --shot 1 --query 5 --eval_query 15 --lr 0.002 --lr_scheduler cosine  --gpu 0 --eval_interval 2 --similarity cosine 
## Acknowledgment
We thank the following repos providing helpful components/functions in our work.
- [ProtoNet](https://github.com/cyvius96/prototypical-network-pytorch)
- [TSP-Head])(https://github.com/hanlu-nju/revisiting-UML/)
