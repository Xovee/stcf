# STCF: Spatial-Temporal Contrasting for Fine-Grained Urban Flow Inference 

![](https://img.shields.io/badge/Under_Review-2021-blue)
![](https://img.shields.io/badge/python-3.8-green)
![](https://img.shields.io/badge/tensorflow-2.5-green)
![](https://img.shields.io/badge/cudatoolkit-11.3-green)
![](https://img.shields.io/badge/cudnn-8.2-green)

![teaser](./img/teaser.gif)

**Left**: coarse-grained taxi flow map; 
**Right**: inferred fine-grained taxi flow map by STCF. 

This repo provides a reference implementation of **STCF** framework described in the following paper:
> **Spatial-Temporal Contrasting for Fine-Grained Urban Flow Inference**  
> Xovee Xu, Ting Zhong, Qiang Gao, Fan Zhou, and Goce Trajcevski  
> Under Review, 2022

## Data

We use [TaxiBJ](https://github.com/yoshall/UrbanFM) P1-P4 and [BikeNYC](https://www.ijcai.org/proceedings/2020/180) datasets.

You can download all five datasets at:

- [Google Drive](https://drive.google.com/drive/folders/1_YgQfrNVrJzsyoTPvu1uhV40tnpuBYVK?usp=sharing) 
- [Baidu Drive](https://pan.baidu.com/s/1r4G4xYtAdamcBaO3V-S01w)  (password: `ndep`)

## Environment

Our experiments are conducted on Ubuntu 20.04, a single NVIDIA 1080Ti GPU, 64GB RAM, and Intel i7 8700K CPU. 
STCF is implemented by `Python 3.8`, `TensorFlow 2.5`, `cudatoolkit 11.3`, and `cudnn 8.2`.
Note: above environment requires NVIDIA driver version `>450.80.02`. 

Create a virtual environment and install GPU-support packages via [Anaconda](https://www.anaconda.com/):
```shell
# create virtual environment
conda create --name=stcf python=3.8 cudatoolkit=11.3 cudnn=8.2
conda activate stcf

# install other dependencies
pip install -r requirements.txt
```

## Usage

**Step 1**: Pre-train the spatial- and temporal-contrasting networks:
```shell
# spatial
python code/sc_pretrain.py --dataset taxi-bj/p1 --note sc 
# temporal
python code/tc_temporal.py --dataset taxi-bj/p1 --note tc  
```

**Step 2**: Fine-tune the coupled network and evaluate performance:
```shell
# fine-tune
python code/stcf.py --dataset taxi-bj/p1 --sc sc --tc tc --model stcf
# evaluate
python code/evaluate.py --dataset taxi-bj/p1 --model stcf
```

More options are described in the code. 

## Cite

If you find our paper & code are helpful for your research, 
please consider citing us :heart_decoration:

```bibtex
@article{xu2022stcf, 
  author = {Xovee Xu and Ting Zhong and Qiang Gao and Fan Zhou and Goce Trajcevski}, 
  title = {Spatial-Temporal Contrasting for Fine-Grained Urban Flow Inference}, 
  journal = {Under review}, 
  year = {2022}, 
}
```

## Acknowledgment

We are particularly grateful for the assistance given by Yuhao Liang and Ce Li. 
We would like to show our gratitude to the authors of UrbanFM, 
FODE, UrbanODE, and others, for sharing their data and codes. 
We express our gratitude to anonymous reviewers for giving constructive feedbacks.

## Contact

For any questions (as well as request for `.pdf` of the paper), 
please open an issue or drop an email to: `xovee at ieee.org`
