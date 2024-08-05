# feature-splatting-ns

Official Nerfstudio implementation of Feature Splatting.

**Note:** Nerfstudio version is designed to be easy-to-use and efficient, which is done via several
tradeoffs than the original feature splatting paper, such as replacing SAM with MobileSAMV2 and
using simple bbox to select Gaussians for editing. We recommend using this repo to check the quality
of features. To reproduce full physics effects and examples on the website, please check out our
[original codebase based on INRIA 3DGS](https://github.com/vuer-ai/feature-splatting-inria).

## Instructions

Follow the [NerfStudio instllation instructions](https://docs.nerf.studio/quickstart/installation.html) to install a conda environment. For convenience,
here are the commands I run to install nerfstudio on two machines.

```bash
# Create an isolated conda environment
conda create --name feature_splatting_ns -y python=3.8
conda activate feature_splatting_ns

# Install necessary NS dependencies
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Insatll nerfstudio
pip install nerfstudio
```

As of this version (0.0.2), we use the gsplat kernel, which comes with NS installation. If you just want to try out feature splatting,
you can run,

```bash
pip install git+https://github.com/vuer-ai/feature-splatting-ns
```

or, for dev purpose, run,

```bash
# Clone and cd to this repository
pip install -e .
```

After this is done, you can train feature-splatting on any nerfstudio-format datasets. An example command is given here,

```bash
ns-download-data nerfstudio --capture-name=poster
ns-train feature-splatting --data data/nerfstudio/poster
```

Specifically, check out various custom outputs defined by nerfstudio under `Output Type`. The `consistent_latent_pca` is used to
project high-dimensional features to low dimensions without flickering effects. After any text is supplied to `Positive Text Queries`,
a new output, `similarity`, will show up in the `Output Type` dropdown menu, which visualizes heatmap response to the language queries.

**NOTE:** Please **PAUSE TRAINING** before using any editing utility. Otherwise it seems to lead to race conditions. Unfortunately fix of this
issue seems to require modifying the core component of NerfStudio, which can not be gracefully implemented as a part of the extension plugin.

### TODOs

- Remove the simple bbox selection and implement better segmentation
- Support more feature extractors
- Improve the object segmentation workflow. Sometimes it causes unexpected error that only prints out in the terminal
- Add estimated gravity / ground plane estimation
- Improve thread safety that seems to lead to race condition when editing / training happen together.
