---
layout: post
title:  "Segformer B0 Demonstrates Powerful and Lightweight Multitask Performance"
summary: "With inspiring efficiency, the Segformer B0 gets competitive scores on joint semantic segmentation and depth estimation."
author: natecibik
date: '2023-09-27 12:27:00 +0530'
category: Segformer
thumbnail: /assets/img/posts/segformer_header.png
keywords: python, monocular, depth, autonomous, vehicles, computer vision, deep learning, machine learning, semantic, segmentation
permalink: /blog/segformer-demonstrates-powerful-multitask-performance/
usemathjax: true
---

The application of transformers to computer vision tasks is a relatively recent phenomenon [[1](https://arxiv.org/abs/2010.11929)], but is now a rapidly developing area of research likely to continue gaining popularity thanks to compelling performance, improving architectures, and computational innovations which avoid the $O(n^2)$ complexity of conventional self-attention [[2](https://arxiv.org/abs/2010.04159), [3](https://arxiv.org/abs/2102.12122)]. Self-attention enables transformers to capture global dependencies in a way that CNNs traditionally lack, as evidenced by previous efforts to incorporate self-attention into CNNs [[4](https://arxiv.org/abs/1711.07971), [5](https://arxiv.org/abs/1906.05909)] or increase their receptive field through dilation [[6](https://arxiv.org/abs/1606.00915), [7](https://arxiv.org/abs/1511.07122)]. This makes transformers seem like a natural choice for vision tasks in domains where global context carries a lot of semantic and geometric meaning, such as in autonomous navigation, and perhaps soon they will challenge the dominance of CNNs in perception stacks.

[Segformer](https://arxiv.org/abs/2105.15203) may be seen as a harbinger of this revolution. Demonstrating impressive performance in semantic segmentation with lightweight designs in both the encoder and decoder which simultaneously improve model efficiency and flexibility, and utilizing the efficient self-attention mechanism from [[3](https://arxiv.org/abs/2102.12122)] which reduces self-attention complexity to $O(\frac{n^2}{R})$, Segformer is an attractive option for semantic segmentation that provides the benefits of global dependencies and contextual awareness offered by vision transformers while avoiding their previous weaknesses, and even offers a range of six model sizes (B0-B5) to choose from based on the application and desired accuracy/efficiency balance.

Making things even more exciting, the [Global-Local Path Network (GLPN)](https://arxiv.org/abs/2201.07436) authors showed that the same hierarchical transformer encoder structure from Segformer could be combined with an efficient decoding head for monocular depth estimation to produce models with head-turning performance and generalization ability. This means it is a relatively simple surgery to construct a multi-task Segformer model which can simultaneously predict semantic segmentation and depth, so I set about doing this for my recent work with [CARLA](https://carla.org/) and the [SHIFT dataset](https://www.vis.xyz/shift/), and the results are impressive.

Since one of the objectives of the project is to keep things as light as possible, I chose the B0 (smallest) version of the Segformer architecture, which has only 3.7 million parameters. First, it was trained for 95,000 training steps (5 epochs) on semantic segmentation only with a linear learning rate schedule from 6e-5 to zero using the front camera of the SHIFT dataset's discrete/images training set. Then, the GLPN depth head was added and the model was trained for another 5 epochs on both tasks with a starting learning rate of 5e-5, showing mutual performance increase in both tasks. After further fine-tuning, the final evaluation scores were a mean IoU of 0.828 and a SiLog loss of 3.07. For a full breakdown of the training process and results, refer to [the full Weights & Biases report](https://api.wandb.ai/links/indezera/4ua2bsyk) attached to the bottom of this post. The code is available on [GitHub](https://github.com/FoamoftheSea/shift-experiments/tree/main). 

Below are videos of the model performing in various driving conditions offered by the dataset. We can see that in the easier daytime example, the performance is very strong with little error, but as the operational domain gets more challenging with less light and more adverse weather, the performance decreases. However, even in the most challenging settings where the video is difficult to parse even for the human eye, the model is mostly able to capture the scene structure, which can likely be attributed in part to the spatial reasoning capabilities of transformers. Keeping in mind that there are just 4.07M parameters jointly estimating two tasks on full resolution images, these results look quite good, and using a larger Segformer backbone would lead to even better results.

<br>

### Inference Videos

Clear weather scene:
<iframe width="990" height="215" src="https://www.youtube.com/embed/iRfZT7b_aCk?si=0NEMHqftfHIVOKK2&amp;showinfo=0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

Overcast scene:
<iframe width="990" height="215" src="https://www.youtube.com/embed/7wUZuGNklXY?si=Q-6Whm7jbTfOtQQR&amp;showinfo=0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

Rain and fog scene:
<iframe width="990" height="215" src="https://www.youtube.com/embed/NtfpmOUL82U?si=9573pGseSJJVbfhZ&amp;showinfo=0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

Rainy night scene:
<iframe width="990" height="215" src="https://www.youtube.com/embed/6N5GgZJ9rRw?si=Ui6VoIDhZQt_VeqA&amp;showinfo=0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

Foggy forest scene:
<iframe width="990" height="215" src="https://www.youtube.com/embed/SJy6R6_cS3Q?si=w1dPkYxdXcBZsPgD&amp;showinfo=0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<br>

### Project Report

<iframe src="https://wandb.ai/indezera/shift-segformer/reports/SHIFT-Multitask-Segformer--Vmlldzo1NTE3MzU3" style="border:none;height:1024px;width:100%"></iframe>