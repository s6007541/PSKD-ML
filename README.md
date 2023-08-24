[![pytorch](https://img.shields.io/badge/pytorch-1.6.0-%2523ee4c2c.svg)](https://pytorch.org/)

# Progressive Self-Knowledge Distillation with Mutual Learning
* PyTorch implementation of **Progressive Self-Knowledge Distillation with Mutual Learning**.  
[[`Slides`](https://docs.google.com/presentation/d/1V0aSEbBalg8lnKeg6NVcS9FPs3CAw2xy/edit?usp=sharing&ouid=104359145724275927049&rtpof=true&sd=true)] [[`Paper`](https://github.com/s6007541/Progressive-Self-Knowledge-Distillation-with-Mutual-Learning/blob/main/pdf_files/PSKD_ML.pdf)] 

## Abstract
Deep neural networks based on large network architectures are often prone to the over-fitting problem and thus inadequate for generalization. Recent self-knowledge distillation (self-KD) approaches have successfully addressed this issue by regularizing a single network using dark knowledge (e.g., knowledge acquired from wrong predictions). Motivated by the idea of online collaborative learning using a large student cohort, we extend the online self-KD methods by combining the two learning schemes as in real-world learning environments. We seek to mimic the real-world self- and collaborative-learning strategies in deep neural networks for the image classification task, aimed to better predict the classification accuracy with lower computational costs during training. We closely explore the performance of a teacher-free dynamically evolving self-distilled network and verify that our approach on the CIFAR-100 dataset gives sufficient insights into combining self-KD and mutual feature learning.
<p align="center">
<img src="image/overview.png" height=350>
</p>

## Requirements
We have tested the code on the following environments: 
* Python 3.7.7 / Pytorch (>=1.6.0) / torchvision (>=0.7.0)

## Datasets
Currently, only **CIFAR-100** and **Tiny-Imagenet** dataset is supported.

## How to Run
### Single-GPU Training
To train a model on single-GPU, run the command as follows:
```bash
$ python main_new.py --PSKD \
                  --BYOT \
                  --DML \
                  --data_type '<DATASET>' \
                  --classifier_type '<MODEL>' \
                  --BYOT_from_k_block '<Number of Blocks in BlackBone>'\

```




## Experimental Results
### Performance measures
* Top-1 Error / Top-5 Error
* Negative Log Likelihood (NLL)
* Expected Calibration Error (ECE)
* Area Under the Risk-coverage Curve (AURC)

### Results on ResNet-18

| Model + Method                               | Dataset   | Top-1 Error | Top-5 Error | NLL      | ECE (%)  | AURC (1e3)|
|----------------------------------------------|:---------:|:-----------:|:-----------:|:--------:|:--------:|:---------:|
| ResNet-18 (baseline)                         | CIFAR-100 | 21.41       | 5.57        | 0.87     | 5.14     | 56.92     |
| ResNet-18 + BYOT                             | CIFAR-100 | 21.89       | 5.64        | 1.00     | 11.24    | 53.79     |
| ResNet-18 + DMFL                             | CIFAR-100 | 21.61       | 5.45        | 0.87     | 5.31     | 57.13     |
| ResNet-18 + PS-KD                            | CIFAR-100 | **19.94**   | 4.79        | 0.79     | 4.21     | 50.44     |
| ResNet-18 + PS-KD + BYOT                     | CIFAR-100 | 20.67       | 4.66        | 0.78     | 7.42     | 50.91     |
| ResNet-18 + PS-KD + DMFL                     | CIFAR-100 | 20.02       | 4.73        | 0.80     | 4.52     | 50.55     |
| ResNet-18 + DMFL  + BYOT                     | CIFAR-100 | 21.18       | 5.47        | 1.02     | 11.74    | 54.89     |
| ResNet-18 + PS-KD + BYOT + DMFL              | CIFAR-100 | 20.03       | **4.16**    | **0.72** | **4.15** | **49.86** |

### Results on ResNet-50

| Model + Method                               | Dataset   | Top-1 Error | Top-5 Error | NLL      | ECE (%)  | AURC (1e3)|
|----------------------------------------------|:---------:|:-----------:|:-----------:|:--------:|:--------:|:---------:|
| ResNet-50 (baseline)                         | CIFAR-100 | 21.96       | 5.21        | 0.89     | 8.75     | 57.24     |
| ResNet-50 + BYOT                             | CIFAR-100 | 19.03       | 4.38        | 0.94     | 11.51    | 48.32     |
| ResNet-50 + DMFL                             | CIFAR-100 | 20.13       | 4.96        | 0.85     | 8.63     | 51.20     |
| ResNet-50 + PS-KD                            | CIFAR-100 | 20.19       | 4.50        | 0.77     | 4.03     | 50.76     |
| ResNet-50 + PS-KD + BYOT                     | CIFAR-100 | 17.92       | 3.94        | 0.75     | 8.96     | 46.76     |
| ResNet-50 + PS-KD + DMFL                     | CIFAR-100 | 19.12       | 4.43        | 0.73     | **3.51** | 47.57     |
| ResNet-50 + DMFL  + BYOT                     | CIFAR-100 | 19.70       | 4.77        | 0.87     | 9.79     | 48.58     |
| ResNet-50 + PS-KD + BYOT + DMFL              | CIFAR-100 | 17.52       | **3.75**    | **0.71** | 10.00    | **46.25** |


### Results on ResNeXt-18

| Model + Method                               | Dataset   | Top-1 Error | Top-5 Error | NLL      | ECE (%)  | AURC (1e3)|
|----------------------------------------------|:---------:|:-----------:|:-----------:|:--------:|:--------:|:---------:|
| ResNeXt-50 (baseline)                        | CIFAR-100 | 19.25       | 4.48        | 0.81     | 7.49     | 48.13     |
| ResNeXt-50 + BYOT                            | CIFAR-100 | 19.09       | 4.54        | 0.84     | 9.62     | 46.66     |
| ResNeXt-50 + DMFL                            | CIFAR-100 | 19.41       | 4.43        | 0.82     | 7.45     | 47.72     |
| ResNeXt-50 + PS-KD                           | CIFAR-100 | 18.79       | 4.20        | 0.74     | **4.85** | 48.52     |
| ResNeXt-50 + PS-KD + BYOT                    | CIFAR-100 | 18.36       | 4.34        | 0.80     | 9.99     | 46.50     |
| ResNeXt-50 + PS-KD + DMFL                    | CIFAR-100 | 19.25       | 4.14        | 0.75     | 5.18     | 47.46     |
| ResNeXt-50 + DMFL  + BYOT                    | CIFAR-100 | 19.30       | 4.23        | 0.88     | 9.86     | 47.98     |
| ResNeXt-50 + PS-KD + BYOT + DMFL             | CIFAR-100 | **18.27**   | **4.11**    | **0.72** | 10.32    | **45.71** |

