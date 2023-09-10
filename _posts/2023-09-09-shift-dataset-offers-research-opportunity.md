---
layout: post
title:  "SHIFT Dataset Offers Research Opportunity in CARLA"
summary: "Generated using the CARLA simulator, the SHIFT dataset can train perception models ready for deployment in the simulator."
author: natecibik
date: '2023-09-09 14:35:23 +0530'
category: SHIFT
thumbnail: /assets/img/posts/shift_banner.jpg
keywords: shift, carla, python, autonomous, computer vision, synthetic data, deep learning, machine learning
permalink: /blog/shift-dataset-offers-research-opportunity/
usemathjax: true
---

In 2022, the Visual Intelligence and Systems Group at ETH Zürich released a large scale synthetic driving dataset called [SHIFT](https://www.vis.xyz/shift/). The dataset was collected using proedural scenario generation and the CARLA simulator, and represents the largest open source synthetic driving dataset at the time of this writing. Motivated by the lack of domain and annotation diversity offered by extant open source driving datasets, SHIFT was released with the stated intention of enabling research in adaptation strategies for imroved model robustness against operational domain shift in agent distributions, time of day, and weather. The dataset offers 5,250 sequences containing 500 frames of 10hz driving data, with metadata descibing the domain. Altogether this makes 2.5 million annotated timesteps, and considering there are 5 cameras in the sensor rig, SHIFT earns its title as the largest synthetic driving dataset available to the public.

[In their paper](https://arxiv.org/abs/2206.08367), the authors exhaustively investigate the negative impact of discrete and continuous domain shift on model performance and adaptation strategies to mitigate it using their new domain shift-inspired synthetic dataset. However, there is a substantially exciting use case offered by the dataset that is not mentioned, ironically related to the operational domain that it is captured in: CARLA. For an engineer interested in training a perception stack to work well within CARLA, SHIFT offers a massive trove of diverse, labeled, in-domain training data that can produce an expert model without even opening the CARLA simulator.

Recently, I've been interested in experimenting with building a lightweight driving stack to explore Reinforcement Learning in CARLA, so that engineer happens to be me. In the series of blog posts to come, I will be documenting my efforts to contruct a sophisticated yet lightweight driving stack to work with the CARLA simulator consisting of perception, planning, and control modules, as well as experimenting with the development of a Reinforcement Learning policy in the planning stack, all within Python. This journey is sure to come with many obstacles and hard-earned wisdom, so I think it's appropriate to document the experience, and share these moments of realization as they come.
