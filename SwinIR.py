import os
import torch
import numpy as np
from tqdm.autonotebook import tqdm

from .network_swinir import SwinIR as SwinIR_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SwinIR_WEIGHTS_URL = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0'


class SwinIR_SR:
    def __init__(self, model_type: str):
        self.model_type = model_type

        model_types = ['classical_sr', 'lightweight', 'real_sr']
        assert model_type in model_types, f'unknown model_type, please choose from: {model_types}'

        os.makedirs(self.weights_folder, exist_ok=True)

        # initialized when calling self.define_model()
        self.model = None
        self.scale = None

    def _download_model_weights(self):
        """downloads the pre-trained weights from GitHub model zoo."""
        if not os.path.exists(self.weights_path):
            os.system(f'wget {SwinIR_WEIGHTS_URL}/{self.weights_name} -P {self.weights_folder}')
            print(f'downloading weights to {self.weights_path}')

    def _load_raw_model(self):
        model_hyperparams = {'upscale': self.scale, 'in_chans': 3, 'img_size': 64, 'window_size': 8,
                             'img_range': 1., 'mlp_ratio': 2, 'resi_connection': '1conv'}

        if self.model_type == 'classical_sr':
            model = SwinIR_model(depths=[6] * 6, embed_dim=180, num_heads=[6] * 6,
                                 upsampler='pixelshuffle', **model_hyperparams)

        elif self.model_type == 'lightweight':
            model = SwinIR_model(depths=[6] * 4, embed_dim=60, num_heads=[6] * 4,
                                 upsampler='pixelshuffledirect', **model_hyperparams)

        elif self.model_type == 'real_sr':
            model = SwinIR_model(depths=[6] * 6, embed_dim=180, num_heads=[6] * 6,
                                 upsampler='nearest+conv', **model_hyperparams)

        return model

    def _load_pretrained_weights(self):
        self._download_model_weights()
        pretrained_weights = torch.load(f'{self.weights_folder}/{self.weights_name}')

        if self.model_type == 'classical_sr':
            return pretrained_weights['params']

        elif self.model_type == 'lightweight':
            return pretrained_weights['params']

        elif self.model_type == 'real_sr':
            return pretrained_weights['params_ema']

    @staticmethod
    def _process_img_for_model(img: np.array) -> torch.tensor:
        """cv2 format - np.array HWC-BGR -> model format torch.tensor NCHW-RGB. (from the official repo)"""
        # (from the official repo)
        img = img.astype(np.float32) / 255.  # image to HWC-BGR, float32
        img = np.transpose(img if img.shape[2] == 1 else img[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img = torch.from_numpy(img).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB
        return img

    @staticmethod
    def _pad_img_for_model(img: torch.tensor, window_size=8) -> torch.tensor:
        """pad input image to be a multiple of window_size (pretrained with window_size=8). (from the official repo)"""
        _, _, h_old, w_old = img.size()
        h_new = (h_old // window_size + 1) * window_size
        w_new = (w_old // window_size + 1) * window_size
        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_new, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_new]
        return img

    def define_model(self, scale: int):
        assert scale in self.scales, 'unsupported scale for this model_type'
        self.scale = scale

        pretrained_weights = self._load_pretrained_weights()
        model = self._load_raw_model()
        model.load_state_dict(pretrained_weights, strict=True)
        model.eval()

        self.model = model.to(device)

    def upscale(self, img: np.array) -> torch.tensor:
        """feed the given image to the super resolution model."""
        h_in, w_in, _ = img.shape
        h_out, w_out = h_in * self.scale, w_in * self.scale

        with torch.no_grad():
            img = self._process_img_for_model(img)
            img = self._pad_img_for_model(img)
            img_upscale = self.model(img)

        return img_upscale[..., :h_out, :w_out]

    @staticmethod
    def model_output_to_numpy(output: torch.tensor) -> np.array:
        """convert the output of the SR model to cv2 format np.array. (from the official repo)"""
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        return output

    def upscale_using_patches(self, img_lq, slice_dim=256, slice_overlap=0, keep_pbar=False):
        """Apply super resolution on smaller patches and return full image"""
        scale = self.scale
        h, w, c = img_lq.shape
        img_hq = np.zeros((h * scale, w * scale, c))

        slice_step = slice_dim - slice_overlap
        num_patches = int(np.ceil(h / slice_step) * np.ceil(w / slice_step))
        with tqdm(total=num_patches, unit='patch', desc='Performing SR on patches', leave=keep_pbar) as pbar:
            for h_slice in range(0, h, slice_step):
                for w_slice in range(0, w, slice_step):
                    h_max = min(h_slice + slice_dim, h)
                    w_max = min(w_slice + slice_dim, w)
                    pbar.set_postfix(Status=f'[{h_slice:4d}-{h_max:4d}, {w_slice:4d}-{w_max:4d}]')

                    # apply super resolution on slice
                    img_slice = img_lq[h_slice:h_max, w_slice:w_max]
                    sr_output = self.upscale(img_slice)
                    img_slice_hq = self.model_output_to_numpy(sr_output)

                    # update full image
                    img_hq[h_slice * scale:h_max * scale, w_slice * scale:w_max * scale] = img_slice_hq
                    pbar.update(1)

            pbar.set_postfix(Status='Done')

        return img_hq

    @property
    def weights_name(self):
        if self.model_type == 'classical_sr':
            return f'001_classicalSR_DF2K_s64w8_SwinIR-M_x{self.scale}.pth'

        elif self.model_type == 'lightweight':
            return f'002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{self.scale}.pth'

        elif self.model_type == 'real_sr':
            return f'003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'

    @property
    def weights_folder(self):
        return f'{os.path.dirname(__file__)}/weights'

    @property
    def weights_path(self):
        return f'{self.weights_folder}/{self.weights_name}'

    @property
    def scales(self):
        if self.model_type == 'classical_sr':
            return [2, 3, 4, 8]

        elif self.model_type == 'lightweight':
            return [2, 3, 4]

        elif self.model_type == 'real_sr':
            return [4]
