# ETIP-Problem

**ETIP** is a general element tagging problem, which allows not only the elements could be a phrase, clause or sentence, but also a phrase or clause element could be embedded in sentence element.

-------------------

[TOC]

## Word Segmentation Method

> [Jieba](https://github.com/fxsjy/jieba)
> [NLPIR](https://github.com/NLPIR-team/NLPIR)

## ETIP Architecture
![Alt text](https://github.com/ETIP-team/ETIP-Project/raw/master/md_imgs/ETIP_architecture.png)



## FlowChart
![Alt text](https://github.com/ETIP-team/ETIP-Project/raw/master/md_imgs/flowchart.png)

## Comparison with other published models in F1 score
| \                 |  C  |  WP | PC  |CP   |IA   |E    |T    |Overall|
| :---------------: |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:-----:|
| CNN               |0.0      |0.722    |0.134    |**0.853**|**0.905**|0.150    |0.844    |0.767
| Layered-BiLSTM+CRF|0.553    |**0.789**|0.623    |0.822    |0.898    |0.556    |0.818    |0.792
| Mention HyperGraph|0.584    |0.600    |0.588    |0.798    |0.840    |0.500    |0.768    |0.761
| CNN-SW            |**0.872**|0.722    |**0.724**|0.846    |0.860    |**0.750**|**0.948**|**0.855**
