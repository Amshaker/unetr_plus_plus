
[Abdelrahman Shaker](https://scholar.google.com/citations?hl=en&user=eEz4Wu4AAAAJ), [Muhammad Maaz](https://scholar.google.com/citations?user=vTy9Te8AAAAJ&hl=en&authuser=1&oi=sra), [Hanoona Rasheed](https://scholar.google.com/citations?user=yhDdEuEAAAAJ&hl=en&oi=ao), [Salman Khan](https://salman-h-khan.github.io/), [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en&oi=ao) and [Fahad Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en)


<br>

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;  [Paper](https://ieeexplore.ieee.org/document/10526382): [<img height="25" src="/unetr_plus_plus/Paper.PNG" width="25" />](https://ieeexplore.ieee.org/document/10526382)  , [Code:](https://github.com/Amshaker/unetr_plus_plus) [<img height="25" src="/unetr_plus_plus/Github.png" width="25" />](https://github.com/Amshaker/unetr_plus_plus)

<br>

## :rocket: News
* **(May 04, 2024):** We're thrilled to share that UNETR++ has been accepted to IEEE TMI-2024! 🎊.
* **(Jun 01, 2023):** UNETR++ code & weights are released for Decathlon-Lung and BRaTs.
* **(Dec 15, 2022):** UNETR++ weights are released for Synapse & ACDC datasets.
* **(Dec 09, 2022):** UNETR++ training and evaluation codes are released.
  
## Overview
<p align="center">
<!-- ![results](images/madds_vs_top_1.png) -->
<img src="images/intro_fig.jpg" width="800">
</p>
<p align="justify" style="width:740px;">
We propose an efficient hybrid hierarchical architecture for 3D medical image segmentation, named UNETR++, that strives to achieve both better segmentation
accuracy and efficiency in terms of parameters and FLOPs. Compared to best existing nnFormer, UNETR++ achieves better segmentation performance while significantly reduces the model complexity by over 71%.
</p>

## Abstract


![main figure](images/UNETR++_Block_Diagram.jpg)

<p align="justify" style="width:740px;">
Owing to the success of transformer models, recent works study their applicability in 3D medical segmentation tasks. 
Within the transformer models, the self-attention mechanism is one of the main building blocks that strives to capture long-range dependencies. However, the self-attention operation has quadratic complexity which proves to be a computational bottleneck, especially in volumetric medical imaging, where the inputs are 3D with numerous slices.
    
In this paper, we propose a 3D medical image segmentation approach, named UNETR++, that offers both high-quality segmentation masks as well as efficiency in terms of parameters, compute cost, and inference speed. The core of our design is the introduction of a novel efficient paired attention (EPA) block that efficiently learns spatial and channel-wise discriminative features using a pair of inter-dependent branches based on spatial and channel attention.
Our spatial attention formulation is efficient having linear complexity with respect to the input sequence length. To enable communication between spatial and channel-focused branches, we share the weights of query and key mapping functions that provide a complimentary benefit (paired attention), while also reducing the overall network parameters. Our extensive evaluations on five benchmarks, Synapse, BTCV, ACDC, BRaTs, and Decathlon-Lung, reveal the effectiveness of our contributions in terms of both efficiency and accuracy. On Synapse, our UNETR++ sets a new state-of-the-art with a Dice Score of 87.2%, while being significantly efficient with a reduction of over 71% in terms of both parameters and FLOPs, compared to the best method in the literature.
</p>


## Comparison with SOTA (Synapse)
![results](images/Synapse_quantitative_results.png)

## Comparison with SOTA (ACDC)
![results](images/Acdc_quantitative_results.png)

## Comparison with SOTA (BTCV)
![results](images/BTCV_quantitative_results.png)

## Comparison with SOTA (BRaTs)
![results](images/Brats_quantitative_results.png)


## Qualitative Baseline Comparison (Synapse)
![results](images/Baseline_Comparison_fig_2.jpg)
![results](images/Baseline_comparison_fig.jpg)

## Qualitative Results (Synapse)

![results](images/UNETR++_results_fig.jpg)

![results](images/Synapse_res_2.png)

## Qualitative Results (ACDC)

![results](images/acdc_vs_unetr_suppl.jpg)


## Citation
If you use our work, please consider citing:
```bibtex
@ARTICLE{10526382,
  title={UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation}, 
  author={Shaker, Abdelrahman M. and Maaz, Muhammad and Rasheed, Hanoona and Khan, Salman and Yang, Ming-Hsuan and Khan, Fahad Shahbaz},
  journal={IEEE Transactions on Medical Imaging}, 
  year={2024},
  doi={10.1109/TMI.2024.3398728}}

```

## Contact
Should you have any question, please create an issue on this repository or contact me at abdelrahman.youssief@mbzuai.ac.ae.
