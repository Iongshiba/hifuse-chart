# TriFuse: a Feature Fusion approach for Chart Detection

## Setup Environment

- To set up the environment for this project, you can create a Conda environment using the following command:

```bash
conda env create -f environment.yml
```

- Replace <env> with your desired environment name, and <this file> with the path to the environment file that lists the required dependencies.

## Method

### Backbone

We utilize the fusion feature of HiFuse architecture to fuse Global features and Local features:

- Global Block: utilize the parallel computing and temporal feature extracting of Transformer blocks to capture long distance dependencies, hence the name Global.
- Local Block: utilize the inductive bias of Convolution blocks to extract small features that are "invisible" to Transformer.
- HFF Block: fuse features map from Global and Local blocks.

### Neck

Feature Pyramid Network is used to further enhance the fused feature maps output by the HiFuse backbone.

### Head

End-to-end Object Detection has recently received its attention due to the emergence of DETR. Therefore, we hypothesize that the Transformer Encoder in the HiFuse Backbone would cooporate well with the Transformer Decoder inside the DETR Head.

- With One-to-one mapping prediction, Non Maximum Suppression is discarded.

## Reference

- [HiFuse](https://arxiv.org/abs/2209.10218)
- [Vision Transformer](https://arxiv.org/abs/2010.11929)
- [Feature Pyramid Network](https://arxiv.org/abs/1612.03144)
- [ViT-UperNet](https://link.springer.com/article/10.1007/s40747-024-01359-6)
- [DETR](https://arxiv.org/abs/2005.12872)
