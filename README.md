# Wrapper for SwinIR

> Based on [__this repository__](https://github.com/JingyunLiang/SwinIR) - the official PyTorch implementation of
> [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257).


* `SwinIR.py` is a minimal wrapper for the super resolution model, making it easy to use as a part from a bigger pipeline.
* Only enables the usage of the pretrained weights from the model zoo. For training, see the official repo.
* Uses the PyTorch model [`network_swinir.py`](https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py) from the official repo (unchanged).

## How to Use

* For easy usage on your own data, see [Demo.ipynb](https://github.com/Lin-Sinorodin/SwinIR_wrapper/blob/main/Demo.ipynb). This notebook will be displayed best using Google Colab which supports more interactive usage:


<div align="center">
<a href="https://colab.research.google.com/github/Lin-Sinorodin/SwinIR_wrapper/blob/main/Demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>


* As a quick preview, this example demonstrates usage with only few lines:
  ```python
  import cv2
  from SwinIR import SwinIR_SR

  # load low quality image
  img_lq = cv2.imread(path, cv2.IMREAD_COLOR)

  # initialize super resolution model
  sr = SwinIR_SR(model_type='real_sr')
  sr.define_model(scale=4)

  # feed the image to the SR model
  img_hq = sr.upscale(img_lq)
  ```

## License and Acknowledgement
Please follow [the license](https://github.com/JingyunLiang/SwinIR#license-and-acknowledgement) of the official repo of this paper. Thanks for their great work! 
