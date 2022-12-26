# UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation

[Abdelrahman Shaker](https://scholar.google.com/citations?hl=en&user=eEz4Wu4AAAAJ), [Muhammad Maaz](https://scholar.google.com/citations?user=vTy9Te8AAAAJ&hl=en&authuser=1&oi=sra), [Hanoona Rasheed](https://scholar.google.com/citations?user=yhDdEuEAAAAJ&hl=en&authuser=1&oi=sra), [Salman Khan](https://salman-h-khan.github.io/), [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en) and [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en)

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://amshaker.github.io/unetr_plus_plus/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2212.04497)
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://docs.google.com/presentation/d/1NzBplLi9MzKJzS1YBc-bX4z-VRyZS1_p/edit?usp=sharing&ouid=117495521772794913474&rtpof=true&sd=true)


## :rocket: News
* **(Dec 15, 2022):** UNETR++ weights are released for Synapse & ACDC datasets.
* **(Dec 09, 2022):** UNETR++ training and evaluation codes are released.

<hr />

![main figure](media/intro_fig.jpg)
> **Abstract:** *Owing to the success of transformer models, recent works study their applicability in 3D medical segmentation tasks. 
Within the transformer models, the self-attention mechanism is one of the main building blocks that strives to capture long-range dependencies, compared to the local convolutional-based design.
However, the self-attention operation has quadratic complexity which proves to be a computational bottleneck, especially in volumetric medical imaging, where the inputs are 3D with numerous slices. 
In this paper, we propose a 3D medical image segmentation approach, named UNETR++, that offers both high-quality segmentation masks as well as efficiency in terms of parameters and compute cost. The core of our design is the introduction of a novel efficient paired attention (EPA) block that efficiently learns spatial and channel-wise discriminative features using a pair of inter-dependent branches based on spatial and channel attention.
Our spatial attention formulation is efficient having linear complexity with respect to the input sequence length. To enable communication between spatial and channel-focused branches, we share the weights of query and key mapping functions that provide a complimentary benefit (paired attention), while also reducing the overall network parameters. Our extensive evaluations on three benchmarks, Synapse, BTCV and ACDC, reveal the effectiveness of the proposed contributions in terms of both efficiency and accuracy. On Synapse dataset, our UNETR++ sets a new state-of-the-art with a Dice Similarity Score of 87.2\%, while being significantly efficient with a reduction of over 71\% in terms of both parameters and FLOPs, compared to the best existing method in the literature.* 
<hr />


## Architecture overview of UNETR++
Overview of our UNETR++ approach with hierarchical encoder-decoder structure. The 3D patches are fed to the encoder, whose outputs are then connected to the decoder via skip connections followed by convolutional blocks to produce the final segmentation mask. The focus of our design is the introduction of an _efficient paired-attention_ (EPA) block. Each EPA block performs two tasks using parallel attention modules with shared keys-queries and different value layers to efficiently learn enriched spatial-channel feature representations. As illustrated in the EPA block diagram (on the right), the first (top) attention module aggregates the spatial features by a weighted sum of the projected features in a linear manner to compute the spatial attention maps, while the second (bottom) attention module emphasizes the dependencies in the channels and computes the channel attention maps. Finally, the outputs of the two attention modules are fused and passed to convolutional blocks to enhance the feature representation, leading to better segmentation masks.
![Architecture overview](media/UNETR++_Block_Diagram.jpg)

<hr />


## Results

### Synapse Dataset
State-of-the-art comparison on the abdominal multi-organ Synapse dataset. We report both the segmentation performance (DSC, HD95) and model complexity (parameters and FLOPs).
Our proposed UNETR++ achieves favorable segmentation performance against existing methods, while being considerably reducing the model complexity. Best results are in bold. 
Abbreviations stand for: Spl: _spleen_, RKid: _right kidney_, LKid: _left kidney_, Gal: _gallbladder_, Liv: _liver_, Sto: _stomach_, Aor: _aorta_, Pan: _pancreas_. 
Best results are in bold.

![Synapse Results](media/synapse_results.png)

<hr />

## Qualitative Comparison

### Synapse Dataset
Qualitative comparison on multi-organ segmentation task. Here, we compare our UNETR++ with existing methods: UNETR, Swin UNETR, and nnFormer. 
The different abdominal organs are shown in the legend below the examples. Existing methods struggle to correctly segment different organs (marked in red dashed box). 
Our UNETR++ achieves promising segmentation performance by accurately segmenting the organs.
![Synapse Qual Results](media/UNETR++_results_fig_synapse.jpg)

### ACDC Dataset
Qualitative comparison on the ACDC dataset. We compare our UNETR++ with existing methods: UNETR and nnFormer. It is noticeable that the existing methods struggle to correctly segment different organs (marked in red dashed box). Our UNETR++ achieves favorable segmentation performance by accurately segmenting the organs.  Our UNETR++ achieves promising segmentation performance by accurately segmenting the organs.
![ACDC Qual Results](media/acdc_vs_unetr_suppl.jpg)


<hr />

## Installation
The code is tested with PyTorch 1.11.0 and CUDA 11.3. After cloning the repository, follow the below steps for installation,

1. Create and activate conda environment
```shell
conda create --name unetr_pp python=3.8
conda activate unetr_pp
```
2. Install PyTorch and torchvision
```shell
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
3. Install other dependencies
```shell
pip install -r requirements.txt
```
<hr />


## Dataset
We follow the same dataset preprocessing as in [nnFormer](https://github.com/282857341/nnFormer). The raw dataset folders should be organized as follows: 

```
./DATASET/
  ├── unetr_pp_raw/
      ├── unetr_pp_raw_data/
          ├── Task01_ACDC/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
          ├── Task02_Synapse/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
      ├── unetr_pp_cropped_data/
 ```
 
Please refer to [Setting up the datasets](https://github.com/282857341/nnFormer) on nnFormer repository for more details.
Alternatively, you can download the preprocessed dataset for both Synapse and ACDC 
from [this link](https://drive.google.com/file/d/1GybpWhjSkLJttJJ75_0Spk0I7mmGcLS1) and extract it under the project directory.

## Training
The following scripts can be used for training our UNETR++ model on Synapse and ACDC datasets,
```shell
bash run_training_synapse.sh
bash run_training_acdc.sh
```

<hr />

## Evaluation

To reproduce the results of UNETR++: 

1- Download [Synapse weights](https://drive.google.com/file/d/1JEOj-DSIs0NqnCwoMHNBImmuGGYx3M5v) and paste it in the following path
```shell
unetr_pp/evaluation/unetr_pp_synapse_checkpoint/unetr_pp/3d_fullres/Task002_Synapse/unetr_pp_trainer_synapse__unetr_pp_Plansv2.1/fold_0/
```
2- Download [ACDC weights](https://drive.google.com/file/d/1-MfsfZl5j25TVWC7QwZqO42Bn3UrHBjY) and paste it in the following path
```shell
unetr_pp/evaluation/unetr_pp_acdc_checkpoint/unetr_pp/3d_fullres/Task001_ACDC/unetr_pp_trainer_acdc__unetr_pp_Plansv2.1/fold_0/
```
3- The following scripts can be used for evaluating our UNETR++ model on Synapse and ACDC datasets,
```shell
bash run_evaluation_synapse.sh
bash run_evaluation_acdc.sh
```

<hr />


## Citation
If you use our work, please consider citing:
```bibtex
    @article{Shaker2022UNETR++,
      title={UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation},
      author={Shaker, Abdelrahman and Maaz, Muhammad and Rasheed, Hanoona and Khan, Salman and Yang, Ming-Hsuan and Khan, Fahad Shahbaz},
      journal={arXiv:2212.04497},
      year={2022},
}
```

## Contact
Should you have any question, please create an issue on this repository or contact at abdelrahman.youssief@mbzuai.ac.ae.
