

# BRIXEL

This is the official implementation of BRIXEL (put paper name). Brixel allows the user to prduce high-resolution feature maps using the DINOv3 backbone without requiring large amounts of compute.

[ :scroll: [`Paper`](https://arxiv.org/abs/2508.10104)] [ :newspaper: [`Blog`](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/)] [ :globe_with_meridians: [`Website`](https://ai.meta.com/dinov3/)] [ :book: [`BibTeX`](#citing-dinov3)]

## Overview

Put the overview figure as well as the example figures.

<div align="center">
  <img width="1364" height="1024" alt="market" src="https://github.com/user-attachments/assets/1411f491-988e-49cb-95ae-d03fe6e3c268" />

  <i></em><b>High-resolution dense features.</b><br/>We visualize the cosine similarity maps obtained with DINOv3 output features<br/> between the patches marked with a red cross and all other patches.</i>
</div>

<br/>



## Installation

a )Either just do install from git 

or b) clone the repo and then hit 
pip install -e .

Then Build DeformAttention
cd BRIXEL/src/brixel/dinov3_main/dinov3/eval/segmentation/models/utils/ops
python setup.py build_ext --inplace




## Pretrained models

To use the pretrained models, please first download the weights of the DINOv3 backbones as outlined in the [DINOv3 repo](https://github.com/huggingface/pytorch-image-models/).

Then download the BRIXEL weights here:

ViT models pretrained on web dataset (LVD-1689M):
<table style="margin: auto">
  <thead>
    <tr>
      <th>Model</th>
      <th>Parameters</th>
      <th>Pretraining<br/>Dataset</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/16 distilled </td>
      <td align="right">21M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-S+/16 distilled</td>
      <td align="right">29M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-B/16 distilled</td>
      <td align="right">86M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-L/16 distilled</td>
      <td align="right">300M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-H+/16 distilled</td>
      <td align="right">840M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-7B/16</td>
      <td align="right">6,716M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/">[link]</a></td>
    </tr>
  </tbody>
</table>

### Build the pretrained models

```python
from brixel.models import build_model

dino_weight_path = '/backbone_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
adapter_weight_path = '/saved_models/dinov3_vitb16.pth'
brixel_model = build_model('dinov3_vitb16', dino_weight_path, adapter_weight_path)
```



### Image transforms

Please use the standard DINOv3 image transform for inputs.

```python
import torchvision
from torchvision.transforms import v2

def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])
```

## License

DINOv3 code and model weights are released under the DINOv3 License. See [LICENSE.md](LICENSE.md) for additional details.

## Citation

If you find this repo useful, please consider citing the paper

```
@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.10104},
}
```
 as well as the DINOv3 paper

```
@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.10104},
}
```
