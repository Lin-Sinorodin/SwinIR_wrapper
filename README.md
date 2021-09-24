# Wrapper for SwinIR

> Based on [__this repository__](https://github.com/JingyunLiang/SwinIR) - the official PyTorch implementation of
> [SwinIR: Image Restoration Using Shifted Window Transformer](https://arxiv.org/abs/2108.10257).


* `SwinIR.py` is a minimal wrapper for the super resolution model, making it easy to use as a part from a bigger pipeline.
* Only enables the usage of the pretrained weights from the model zoo. For training, see the official repo.
* Uses the PyTorch model [`network_swinir.py`](https://github.com/JingyunLiang/SwinIR/blob/main/models/network_swinir.py) from the official repo (unchanged).

> This example demonstrates usage with only few lines:
> ```python
> from SwinIR import SwinIR_SR
> 
> # load low quality image
> img_lq = cv2.imread(path, cv2.IMREAD_COLOR)
> 
> # initialize super resolution model
> sr = SwinIR_SR(model_type='real_sr')
> sr.define_model(scale=4)
> 
> # feed the image to the SR model
> sr_output = sr.upscale(img_lq)
> img_hq = sr.model_output_to_numpy(sr_output)
> ```
