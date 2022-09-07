# Registration based Few-Shot Anomaly Detection

This is an official implementation of “Registration based Few-Shot Anomaly Detection” (RegAD) with PyTorch, accepted by ECCV 2022 (Oral).

[Paper Link](https://arxiv.org/abs/2207.07361)

```
@inproceedings{huang2022regad,
  title={Registration based Few-Shot Anomaly Detection}
  author={Huang, Chaoqin and Guan, Haoyan and Jiang, Aofan and Zhang, Ya and Spratlin, Michael and Wang, Yanfeng},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

<center><img src="figures/intro_final.png "width="80%"></center>

**Abstract**:  This paper considers few-shot anomaly detection (FSAD), a practical yet under-studied setting for anomaly detection (AD), where only a limited number of normal images are provided for each category at training. So far, existing FSAD studies follow the one-model-per-category learning paradigm used for standard AD, and the inter-category commonality has not been explored. Inspired by how humans detect anomalies, i.e., comparing an image in question to normal images, we here leverage registration, an image alignment task that is inherently generalizable across categories, as the proxy task, to train a category-agnostic anomaly detection model. During testing, the anomalies are identified by comparing the registered features of the test image and its corresponding support (normal) images. As far as we know, this is the first FSAD method that trains a single generalizable model and requires no re-training or parameter fine-tuning for new categories. 

**Keywords**: Anomaly Detection, Few-Shot Learning, Registration

## Get Started

### Environment
- python >= 3.7.11
- pytorch >= 1.11.0
- torchvision >= 0.12.0
- numpy >= 1.19.5
- scipy >= 1.7.3
- skimage >= 0.19.2
- matplotlib >= 3.5.2
- kornia >= 0.6.5
- tqdm

### Files Preparation

1. Download the MVTec dataset [here](https://www.mvtec.com/company/research/datasets/mvtec-ad). 
2. Download the support dataset for few-shot anomaly detection on [Google Drive](https://drive.google.com/file/d/1AZcc77cmDfkWA8f8cs-j-CUuFFQ7tPoK/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1GZAqtscOaPliaFCiSKlViA) (i9rx)
and unzip the dataset. For those who have problem downloading the support set, please optional download categories of capsule and grid on [Baidu Disk](https://pan.baidu.com/s/1fFwAB__bV0ja38B4w3JnXQ) (pll9) and [Baidu Disk](https://pan.baidu.com/s/1_--hXPPnlv3Tv7HHd4HRZQ) (ns0n).
    ```
    tar -xvf support_set.tar
    ```
    We hope the followers could use these support datasets to make a fair comparison between different methods.
3. Download the pre-train models on [Google Drive](https://drive.google.com/file/d/1guZBh40btPRmxcnY_lud88V1NoT-eWWX/view?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/1w7-6zicbZA6ysHMSpTHNhg) (4qyo)
and unzip the checkpoint files.
    ```
    tar -xvf save_checkpoints.tar
    ```
  After the preparation work, the whole project should have the following structure:

  ```
  ./RegAD
  ├── README.md
  ├── train.py                                  # training code
  ├── test.py                                   # testing code
  ├── MVTec                                     # MVTec dataset files
  │   ├── bottle
  │   ├── cable
  │   ├── ...                  
  │   └── zippper
  ├── support_set                               # MVTec support dataset files
  │   ├── 2
  │   ├── 4                 
  │   └── 8
  ├── models                                    # models and backbones
  │   ├── stn.py  
  │   └── siamese.py
  ├── losses                                    # losses
  │   └── norm_loss.py  
  ├── datasets                                  # dataset                      
  │   └── mvtec.py
  ├── save_checkpoints                          # model checkpoint files                  
  └── utils                                     # utils
      ├── utils.py
      └── funcs.py
  ```

### Quick Start

```python
python test.py --obj $target-object --shot $few-shot-number --stn_mode rotation_scale
```

For example, if run on the category `bottle` with `k=2`:
```python
python test.py --obj bottle --shot 2 --stn_mode rotation_scale
```

## Training

```python
python train.py --obj $target-object --shot $few-shot-number --data_type mvtec --data_path ./MVTec/ --epochs 50 --batch_size 32 --lr 0.0001 --momentum 0.9 --inferences 10 --stn_mode rotation_scale 
```

For example, to train a RegAD model on the MVTec dataset on `bottle` with `k=2`, simply run:

```python
python train.py --obj bottle --shot 2 --data_type mvtec --data_path ./MVTec/ --epochs 50 --batch_size 32 --lr 0.0001 --momentum 0.9 --inferences 10 --stn_mode rotation_scale 
```

Then you can run the evaluation using:
```python
python test.py --obj bottle --shot 2 --stn_mode rotation_scale
```

## Results

Results of few-shot anomaly detection and localization with k=2:

<div style="text-align: center;">
<table>
<tr><td>AUC (%)</td> <td colspan="2">Detection</td> <td colspan="2">Localization</td></tr>
<tr><td>K=2</td> <td>RegAD</td> <td>Inplementation</td> <td>RegAD</td> <td>Inplementation</td></tr>
<tr height='21' style='mso-height-source:userset;height:16pt' id='r0'>
<td height='21' class='x21' width='90' style='height:16pt;width:67.5pt;'>bottle</td>
<td class='x23' width='90' style='width:67.5pt;'>99.4</td>
<td class='x22' width='90' style='width:67.5pt;'><b>99.7</b></td>
<td class='x23' width='90' style='width:67.5pt;'>98.0</td>
<td class='x22' width='90' style='width:67.5pt;'><b>98.6</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r1'>
<td height='21' class='x21' style='height:16pt;'>cable</td>
<td class='x23'>65.1</td>
<td class='x22'><b>69.8</b></td>
<td class='x23'>91.7</td>
<td class='x22'><b>94.2</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r2'>
<td height='21' class='x21' style='height:16pt;'>capsule</td>
<td class='x23'>67.5</td>
<td class='x22'><b>68.6</b></td>
<td class='x23'>97.3</td>
<td class='x22'><b>97.6</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r3'>
<td height='21' class='x21' style='height:16pt;'>carpet</td>
<td class='x23'>96.5</td>
<td class='x22'><b>96.7</b></td>
<td class='x22'><b>98.9</b></td>
<td class='x22'><b>98.9</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r4'>
<td height='21' class='x21' style='height:16pt;'>grid</td>
<td class='x22'><b>84.0</b></td>
<td class='x23'>79.1</td>
<td class='x23'>77.4</td>
<td class='x22'><b>77.5</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r5'>
<td height='21' class='x21' style='height:16pt;'>hazelnut</td>
<td class='x23'>96.0</td>
<td class='x22'><b>96.3</b></td>
<td class='x23'>98.1</td>
<td class='x22'><b>98.2</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r6'>
<td height='21' class='x21' style='height:16pt;'>leather</td>
<td class='x23'>99.4</td>
<td class='x22'><b>100</b></td>
<td class='x23'>98.0</td>
<td class='x22'><b>99.2</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r7'>
<td height='21' class='x21' style='height:16pt;'>metal_nut</td>
<td class='x23'>91.4</td>
<td class='x22'><b>94.2</b></td>
<td class='x23'>96.9</td>
<td class='x22'><b>98.0</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r8'>
<td height='21' class='x21' style='height:16pt;'>pill</td>
<td class='x23'><b>81.3</b></td>
<td class='x23'>66.1</td>
<td class='x23'>93.6</td>
<td class='x22'><b>97.0</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r9'>
<td height='21' class='x21' style='height:16pt;'>screw</td>
<td class='x23'>52.5</td>
<td class='x22'><b>53.9</b></td>
<td class='x22'><b>94.4</b></td>
<td class='x23'>94.1</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r10'>
<td height='21' class='x21' style='height:16pt;'>tile</td>
<td class='x23'>94.3</td>
<td class='x22'><b>98.9</b></td>
<td class='x23'>94.3</td>
<td class='x22'><b>95.1</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r11'>
<td height='21' class='x21' style='height:16pt;'>toothbrush</td>
<td class='x23'>86.6</td>
<td class='x22'><b>86.8</b></td>
<td class='x22'><b>98.2</b></td>
<td class='x22'><b>98.2</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r12'>
<td height='21' class='x21' style='height:16pt;'>transistor</td>
<td class='x22'><b>86.0</b></td>
<td class='x23'>82.2</td>
<td class='x22'><b>93.4</b></td>
<td class='x23'>93.3</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r13'>
<td height='21' class='x21' style='height:16pt;'>wood</td>
<td class='x23'>99.2</td>
<td class='x22'><b>99.8</b></td>
<td class='x23'>93.5</td>
<td class='x22'><b>96.5</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r14'>
<td height='21' class='x21' style='height:16pt;'>zipper</td>
<td class='x23'>86.3</td>
<td class='x22'><b>90.9</b></td>
<td class='x23'>95.1</td>
<td class='x22'><b>98.3</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r15'>
<td height='21' class='x23' style='height:16pt;'>average</td>
<td class='x22' x:fmla="=AVERAGE(B1:B15)"><b>85.7</b></td>
<td class='x23' x:fmla="=AVERAGE(C1:C15)">85.5</td>
<td class='x23' x:fmla="=AVERAGE(D1:D15)">94.6</td>
<td class='x22' x:fmla="=AVERAGE(E1:E15)"><b>95.6</b></td>
 </tr>
</table>
</div>

Results of few-shot anomaly detection and localization with k=4:

<div style="text-align: center;">
<table>
<tr><td>AUC (%)</td> <td colspan="2">Detection</td> <td colspan="2">Localization</td></tr>
<tr><td>K=4</td> <td>RegAD</td> <td>Inplementation</td> <td>RegAD</td> <td>Inplementation</td></tr>
<tr height='21' style='mso-height-source:userset;height:16pt' id='r0'>
<td height='21' class='x21' width='90' style='height:16pt;width:67.5pt;'>bottle</td>
<td class='x22' width='90' style='width:67.5pt;'><b>99.4</b></td>
<td class='x23' width='90' style='width:67.5pt;'>99.3</td>
<td class='x23' width='90' style='width:67.5pt;'>98.4</td>
<td class='x22' width='90' style='width:67.5pt;'><b>98.5</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r1'>
<td height='21' class='x21' style='height:16pt;'>cable</td>
<td class='x23'>76.1</td>
<td class='x22'><b>82.9</b></td>
<td class='x23'>92.7</td>
<td class='x22'><b>95.5</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r2'>
<td height='21' class='x21' style='height:16pt;'>capsule</td>
<td class='x23'>72.4</td>
<td class='x22'><b>77.3</b></td>
<td class='x23'>97.6</td>
<td class='x22'><b>98.3</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r3'>
<td height='21' class='x21' style='height:16pt;'>carpet</td>
<td class='x22'><b>97.9</b></td>
<td class='x22'><b>97.9</b></td>
<td class='x22'><b>98.9</b></td>
<td class='x22'><b>98.9</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r4'>
<td height='21' class='x21' style='height:16pt;'>grid</td>
<td class='x22'><b>91.2</b></td>
<td class='x23'>87</td>
<td class='x22'><b>85.7</b></td>
<td class='x22'><b>85.7</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r5'>
<td height='21' class='x21' style='height:16pt;'>hazelnut</td>
<td class='x23'>95.8</td>
<td class='x22'><b>95.9</b></td>
<td class='x23'>98.0</td>
<td class='x22'><b>98.4</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r6'>
<td height='21' class='x21' style='height:16pt;'>leather</td>
<td class='x22'><b>100</b></td>
<td class='x23'>99.9</td>
<td class='x22'><b>99.1</b></td>
<td class='x23'>99</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r7'>
<td height='21' class='x21' style='height:16pt;'>metal_nut</td>
<td class='x22'><b>94.6</b></td>
<td class='x23'>94.3</td>
<td class='x22'><b>97.8</b></td>
<td class='x23'>96.5</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r8'>
<td height='21' class='x21' style='height:16pt;'>pill</td>
<td class='x22'><b>80.8</b></td>
<td class='x23'>74.0</td>
<td class='x22'><b>97.4</b></td>
<td class='x22'><b>97.4</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r9'>
<td height='21' class='x21' style='height:16pt;'>screw</td>
<td class='x23'>56.6</td>
<td class='x22'><b>59.3</b></td>
<td class='x23'>95.0</td>
<td class='x22'><b>96.0</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r10'>
<td height='21' class='x21' style='height:16pt;'>tile</td>
<td class='x23'>95.5</td>
<td class='x22'><b>98.2</b></td>
<td class='x23'><b>94.9</b></td>
<td class='x23'>92.6</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r11'>
<td height='21' class='x21' style='height:16pt;'>toothbrush</td>
<td class='x23'>90.9</td>
<td class='x22'><b>91.1</b></td>
<td class='x22'><b>98.5</b></td>
<td class='x22'><b>98.5</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r12'>
<td height='21' class='x21' style='height:16pt;'>transistor</td>
<td class='x23'>85.2</td>
<td class='x22'><b>85.5</b></td>
<td class='x22'><b>93.8</b></td>
<td class='x23'>93.5</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r13'>
<td height='21' class='x21' style='height:16pt;'>wood</td>
<td class='x24'>98.6</td>
<td class='x25'><b>98.9</b></td>
<td class='x24'>94.7</td>
<td class='x25'><b>96.3</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r14'>
<td height='21' class='x21' style='height:16pt;'>zipper</td>
<td class='x23'>88.5</td>
<td class='x22'><b>95.8</b></td>
<td class='x23'>94.0</td>
<td class='x22'><b>98.6</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r15'>
<td height='21' class='x23' style='height:16pt;'>average</td>
<td class='x23' x:fmla="=AVERAGE(B1:B15)">88.2</td>
<td class='x22' x:fmla="=AVERAGE(C1:C15)"><b>89.2</b></td>
<td class='x23' x:fmla="=AVERAGE(D1:D15)">95.8</td>
<td class='x22' x:fmla="=AVERAGE(E1:E15)"><b>96.2</b></td>
 </tr></table>
</div>

Results of few-shot anomaly detection and localization with k=8:

<div style="text-align: center;">
<table>
<tr><td>AUC (%)</td> <td colspan="2">Detection</td> <td colspan="2">Localization</td></tr>
<tr><td>K=8</td> <td>RegAD</td> <td>Inplementation</td> <td>RegAD</td> <td>Inplementation</td></tr>
<tr height='21' style='mso-height-source:userset;height:16pt' id='r0'>
<td height='21' class='x21' width='90' style='height:16pt;width:67.5pt;'>bottle</td>
<td class='x22' width='90' style='width:67.5pt;'><b>99.8</b></td>
<td class='x22' width='90' style='width:67.5pt;'><b>99.8</b></td>
<td class='x23' width='90' style='width:67.5pt;'>97.5</td>
<td class='x22' width='90' style='width:67.5pt;'><b>98.5</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r1'>
<td height='21' class='x21' style='height:16pt;'>cable</td>
<td class='x23'>80.6</td>
<td class='x22'><b>81.5</b></td>
<td class='x23'>94.9</td>
<td class='x22'><b>95.8</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r2'>
<td height='21' class='x21' style='height:16pt;'>capsule</td>
<td class='x23'>76.3</td>
<td class='x22'><b>78.4</b></td>
<td class='x23'>98.2</td>
<td class='x22'><b>98.4</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r3'>
<td height='21' class='x21' style='height:16pt;'>carpet</td>
<td class='x23'>98.5</td>
<td class='x22'><b>98.6</b></td>
<td class='x22'><b>98.9</b></td>
<td class='x22'><b>98.9</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r4'>
<td height='21' class='x21' style='height:16pt;'>grid</td>
<td class='x22'><b>91.5</b></td>
<td class='x22'><b>91.5</b></td>
<td class='x22'><b>88.7</b></td>
<td class='x22'><b>88.7</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r5'>
<td height='21' class='x21' style='height:16pt;'>hazelnut</td>
<td class='x23'>96.5</td>
<td class='x22'><b>97.3</b></td>
<td class='x22'><b>98.5</b></td>
<td class='x22'><b>98.5</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r6'>
<td height='21' class='x21' style='height:16pt;'>leather</td>
<td class='x22'><b>100</b></td>
<td class='x22'><b>100</b></td>
<td class='x23'>98.9</td>
<td class='x22'><b>99.3</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r7'>
<td height='21' class='x21' style='height:16pt;'>metal_nut</td>
<td class='x23'>98.3</td>
<td class='x22'><b>98.6</b></td>
<td class='x23'>96.9</td>
<td class='x22'><b>98.3</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r8'>
<td height='21' class='x21' style='height:16pt;'>pill</td>
<td class='x23'><b>80.6</b></td>
<td class='x23'>77.8</td>
<td class='x23'><b>97.8</b></td>
<td class='x23'>97.7</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r9'>
<td height='21' class='x21' style='height:16pt;'>screw</td>
<td class='x23'>63.4</td>
<td class='x22'><b>65.8</b></td>
<td class='x23'>97.1</td>
<td class='x22'><b>97.3</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r10'>
<td height='21' class='x21' style='height:16pt;'>tile</td>
<td class='x23'>97.4</td>
<td class='x22'><b>99.6</b></td>
<td class='x23'>95.2</td>
<td class='x22'><b>96.1</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r11'>
<td height='21' class='x21' style='height:16pt;'>toothbrush</td>
<td class='x22'><b>98.5</b></td>
<td class='x23'>96.6</td>
<td class='x22'>98.7</td>
<td class='x22'><b>99.0</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r12'>
<td height='21' class='x21' style='height:16pt;'>transistor</td>
<td class='x22'><b>93.4</b></td>
<td class='x23'>90.3</td>
<td class='x22'><b>96.8</b></td>
<td class='x23'>95.9</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r13'>
<td height='21' class='x21' style='height:16pt;'>wood</td>
<td class='x23'>99.4</td>
<td class='x22'><b>99.5</b></td>
<td class='x23'>94.6</td>
<td class='x22'><b>96.5</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r14'>
<td height='21' class='x21' style='height:16pt;'>zipper</td>
<td class='x22'><b>94.0</b></td>
<td class='x23'>93.4</td>
<td class='x22'><b>97.4</b></td>
<td class='x22'><b>97.4</b></td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r15'>
<td height='21' class='x23' style='height:16pt;'>average</td>
<td class='x23' x:fmla="=AVERAGE(B1:B15)"><b>91.2</b></td>
<td class='x22' x:fmla="=AVERAGE(C1:C15)"><b>91.2</b></td>
<td class='x23' x:fmla="=AVERAGE(D1:D15)">96.8</td>
<td class='x22' x:fmla="=AVERAGE(E1:E15)"><b>97.1</b></td>
 </tr></table>
</div>

## Visualization
<center><img src="figures/results.png "width="60%"></center>

## Acknowledgement
We borrow some codes from [SimSiam](https://github.com/facebookresearch/simsiam), [STN](https://github.com/YotYot/CalibrationNet/blob/2446a3bcb7ff4aa1e492adcde62a4b10a33635b4/models/configurable_stn_no_stereo.py) and [PaDiM](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master)

## Contact

If you have any problem with this code, please feel free to contact **huangchaoqin@sjtu.edu.cn**.

