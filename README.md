

# BRIXEL

This is the official implementation of <a href="https://arxiv.org/abs/2511.05168">**Another BRIXEL in the Wall: Towards Cheaper Dense Features**</a>. BRIXEL allows the user to produce high-resolution feature maps using the DINOv3 backbone without requiring large amounts of compute.


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

Note that PyTorch is not automatically installed as a dependency, so BRIXEL assumes it to be installed prior. The package has been tested with PyTorch version 2.9.0. Finally, you will need to build Deformable Attention for DINO:
```
cd brixel/dinov3_main/dinov3/eval/segmentation/models/utils/ops
python setup.py build_ext --inplace
```


### b) Clone the repo
If you wish to work with or modify the code, please clone the repo and install the dependencies from requirements.txt, as well as PyTorch.
```
git clone https://github.com/alexanderlappe/BRIXEL.git
pip install -r requirements.txt
pip install -e .
```
To build Deformable Attention for DINO, run the following:
```
cd srcbrixel/dinov3_main/dinov3/eval/segmentation/models/utils/ops # navigate to this directory within the installed package
python setup.py build_ext --inplace
```




## Pretrained models

To use the pretrained models, please first download the weights of the DINOv3 backbones as outlined in the [DINOv3 repo](https://github.com/huggingface/pytorch-image-models/).

Then download the BRIXEL weights here:

<table style="margin: auto">
  <thead>
    <tr>
      <th>Model</th>
      <th>Image Size</th>
      <th>Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S </td>
      <td align="center">256</td>
      <td align="center"><a href="https://drive.google.com/file/d/1ItRulT6xzhkY6DHRJi8t5k3kbDAoLeVf/view?usp=drive_link">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-B </td>
      <td align="center">256</td>
      <td align="center"><a href="https://drive.google.com/file/d/1eVYNr1ZyhxTSPwbUb-aPdD52d2hGubLQ/view?usp=sharing">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-L </td>
      <td align="center">256</td>
      <td align="center"><a href="https://drive.google.com/file/d/1LuRXvqxz_T5xIKe4DANc4KW3hiPUETzX/view?usp=sharing">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-H+ </td>
      <td align="center">256</td>
      <td align="center"><a href="https://drive.google.com/file/d/1h3erb8L1gG_dUMw57mGIL06LGuxHD47n/view?usp=sharing">[link]</a></td>
    </tr>
    <tr>
      <td>ViT-B </td>
      <td align="center">480</td>
      <td align="center"><a href="https://drive.google.com/file/d/1f_vpsCCkGZCXGgoGgd0AnJ1GoQJSvmee/view?usp=sharing">[link]</a></td>
    </tr>
  </tbody>
</table>

### Build the pretrained models

```python
from brixel.models import build_model

dino_weight_path = '/backbone_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
adapter_weight_path = '/saved_models/dinov3_vitb16.pth'
brixel_model = build_model('dinov3_vitb16', dino_weight_path, adapter_weight_path)

# model identifier should be one of ['dinov3_vits16', 'dinov3_vitb16, 'dinov3_vitl16', 'dinov3_vith16plus']
```



### Usage

This snippet loads a model and computes and visualizes dense feature for an example image.

```python
import torch
from torchvision.transforms import v2
import requests
from PIL import Image
from io import BytesIO

from brixel.models import build_model
from brixel.utils.visualize_features import show_img_and_feat

# Use the DINOv3 image transform
def make_transform(resize_size: int = 256):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])

dino_weight_path = '/backbone_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
adapter_weight_path = '/saved_models/dinov3_vitl16.pth'
model = build_model('dinov3_vitl16', dino_weight_path, adapter_weight_path)

url = "https://github.com/alexanderlappe/BRIXEL/blob/master/figs/goliath.jpg?raw=true"
transform = make_transform()

response = requests.get(url)
image = Image.open(BytesIO(response.content))

x = transform(image).unsqueeze(0)
# get dense features of DINOv3
low_res_feats = model.adapter.backbone.get_intermediate_layers(x, n=1, return_class_token=False, reshape=True)[0]
# get BRIXEL dense features
high_res_feats = model(x)
# visualize using PCA like in the paper
show_img_and_feat(x.squeeze(), high_res_feats.squeeze(), high_res_feats.squeeze(), low_res_target=low_res_feats.squeeze())
```

## Examples

<div align="center">
  <img width="1364" height="1024" alt="market" src="https://github.com/alexanderlappe/BRIXEL/blob/master/figs/qualitative.png" />

  <i></em>Examples of dense features maps produced by the BRIXEL models.</i>
</div>

<br/>

## License
BRIXEL itself is licensed under the MIT license. Please note that DINOv3 code and model weights are released under the <a href="https://github.com/alexanderlappe/BRIXEL/blob/master/src/brixel/dinov3_main/LICENSE.md">DINOv3 License</a>.
## Citation

If you find this repo useful, please consider citing the paper:


```
@misc{lappe2025brixelwallcheaperdense,
      title={Another BRIXEL in the Wall: Towards Cheaper Dense Features}, 
      author={Alexander Lappe and Martin A. Giese},
      year={2025},
      eprint={2511.05168},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.05168}, 
}
```

 as well as the DINOv3 paper:

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
