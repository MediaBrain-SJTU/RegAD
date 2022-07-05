# Registration based Few-Shot Anomaly Detection

This is an official implementation of “Registration based Few-Shot Anomaly Detection” (RegAD) with PyTorch, accepted by ECCV 2022.

```
@inproceedings{huang2022regad,
  title={Registration based Few-Shot Anomaly Detection}
  author={Huang, Chaoqin and Guan, Haoyan and Jiang, Aofan and Zhang, Ya and Spratlin, Michael and Wang, Yanfeng },
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

**Abstract**:  This paper considers few-shot anomaly detection (FSAD), a practical yet under-studied setting for anomaly detection (AD), where only a limited number of normal images are provided for each category at training. So far, existing FSAD studies follow the one-model-per-category learning paradigm used for standard AD, and the inter-category commonality has not been explored. Inspired by how humans detect anomalies, i.e., comparing an image in question to normal images, we here leverage registration, an image alignment task that is inherently generalizable across categories, as the proxy task, to train a category-agnostic anomaly detection model. During testing, the anomalies are identified by comparing the registered features of the test image and its corresponding support (normal) images. As far as we know, this is the first FSAD method that trains a single generalizable model and requires no re-training or parameter fine-tuning for new categories. Experimental results have shown that the proposed method outperforms the state-of-the-art FSAD methods by 3%-8% in AUC on the MVTec and MPDD benchmarks.

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
