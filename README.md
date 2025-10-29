

# BRIXEL

This is the official implementation of **Another BRIXEL in the Wall: Cheap Dense Features for DINOv3** (put link). BRIXEL allows the user to produce high-resolution feature maps using the DINOv3 backbone without requiring large amounts of compute.


## Overview
<div align="center">
  <img width="1364" height="1024" alt="market" src="https://github.com/alexanderlappe/BRIXEL/blob/master/figs/overview.png" />

  <i></em>Equipped with BRIXEL, DINOv3 outputs higher resolution features at a fraction of the computational cost.</i>
</div>

<br/>



## Installation
### a) Install as a package
If you just need the pretrained models to generate dense features, you can simply install BRIXEL as a package and build the models as shown below.
```
pip install "git+https://github.com/alexanderlappe/BRIXEL.git"
```

Note that PyTorch is not automatically installed as a dependency, but necessary to run the models. Finally, build Deformable attention:
```
cd BRIXEL/src/brixel/dinov3_main/dinov3/eval/segmentation/models/utils/ops
python setup.py build_ext --inplace
```


### b) Clone the repo
If you wish to work with or modify the code, please clone the repo and install from requirements.txt, as well as PyTorch.
```
git clone # put in the correct command
pip install -r requirements.txt
pip install -e .
```
To build Deformable Attention,
```
cd brixel/dinov3_main/dinov3/eval/segmentation/models/utils/ops # navigate to this directory within the installed package
python setup.py build_ext --inplace
```




## Pretrained models

To use the pretrained models, please first download the weights of the DINOv3 backbones as outlined in the [DINOv3 repo](https://github.com/huggingface/pytorch-image-models/).

Then download the BRIXEL weights here:

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
      <td>ViT-S </td>
      <td align="right">21M</td>
      <td align="center">LVD-1689M</td>
      <td align="center"><a href="https://drive.google.com/file/d/1ItRulT6xzhkY6DHRJi8t5k3kbDAoLeVf/view?usp=drive_link">[link]</a></td>
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

## Examples

<div align="center">
  <img width="1364" height="1024" alt="market" src="https://github.com/alexanderlappe/BRIXEL/blob/master/figs/qualitative.png" />

  <i></em>Examples of dense features maps produced by the BRIXEL models.</i>
</div>

<br/>

## License
BRIXEL itself is licensed under the MIT license. Please note that DINOv3 code and model weights are released under the DINOv3 License. See the original [LICENSE.md](LICENSE.md) for additional details.

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
