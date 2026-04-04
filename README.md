<div align="center">

# Unsupervised Few-shot Food Recognition with Intra-Class Variationand Inter-Class Similarity Modeling (UFFR-IVIS)

**Na Zheng**<sup>1</sup> &nbsp; **Xuemeng Song**<sup>2</sup>✉ &nbsp; **Wai Teng Tang**<sup>3</sup> &nbsp; **See-Kiong Ng**<sup>1</sup> &nbsp; **Liqiang Nie**<sup>4</sup> &nbsp; **Roger Zimmermann**<sup>1</sup>

<sup>1</sup>School of Computing, National University of Singapore
<sup>2</sup>Department of Data Science, Department of Data Science, Southern University of Science and Technology 
<sup>3</sup>GrabTaxi Holdings Pte., Ltd.
<sup>4</sup>Harbin Institute of Technology (Shenzhen)

✉ Corresponding author  
</div>

## 📌 Introduction

This repository contains the official implementation of the paper **Unsupervised Few-shot Food Recognition with Intra-Class Variationand Inter-Class Similarity Modeling**. It focuses on the **Unsupervised Few-shot Learning for Food Recoginition** task: it leverages large-scale unlabeled food data during training to capture intra-class variation and inter-class similarity, and adapts to novel classes at test time using only a few labeled examples.

### Framework Figure

```markdown
![Framework](assets/framework.png)
```

**Figure 1.** Overall framework of `UFFR-IVIS`. It consists of  (1) dual diversity-injected support/query representation learning that introduces instance-level and representation-level diversities for the representation learning of support/query instance to model the characteristics of high intra-class variation; and  (2) dual regularization-enhanced meta learning that designs two regularizations: auxiliary task-based intra-class regularization and similarity-guided inter-class regularization to regularize the intra-class variation and inter-class similarity modeling, respectively.

---

## Project Structure

```text
.
├── assets/                
├── data/                  
├── model/
├── eval.py
├── test.py
├── train.py
├── utils.py                  
├── README.md
├── requirements.txt
└── LICENSE
```

## Dataset

### MiniFood and LargeFood
Downlado data here [https://drive.google.com/drive/folders/1Bz7cygevBJd9_gEKhTbejQqW45m9FEIH?usp=drive_link]

## Training scripts

python train.py --eval_all --unsupervised --batch_size 64 --augment 'ws' - --max_epoch 300 --model_class ProtoNet --backbone_class ConvNet --dataset Food2K --way 5 --shot 1 --query 5 --eval_query 15 --lr 0.002 --lr_scheduler cosine  --gpu 0 --eval_interval 2 --similarity cosine 
## Test scripts
python eval.py --path checkpoint.pth \
  --eval_all --model_class ProtoNet --backbone_class ConvNet --num_test_episodes 1000 \
  --gpu 0 --eval_dataset Food2K --augment test --similarity cosine
## Acknowledgment
We thank the following repos providing helpful components/functions in our work.
- [ProtoNet](https://github.com/cyvius96/prototypical-network-pytorch)
- [TSP-Head])(https://github.com/hanlu-nju/revisiting-UML/)
