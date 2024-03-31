This code is based on this repo with the purpose of completing the Week 7 challange on Founders and Coders MLX3.

Week 7 challenge details
=========================

Diffusion Models
This week, we shift our focus to another exciting area in deep learning: Diffusion Models. These models, particularly in the context of image processing and generation, have shown remarkable capabilities. Your main task will involve understanding and implementing diffusion models, with a specific focus on the Conditional Diffusion Model for MNIST digits.

Goal of the week
Your primary objective for this week is to develop and train a Conditional Diffusion Model. This model should be capable of generating high-quality images of handwritten digits, similar to those found in the MNIST dataset.

To guide you through this task, here is an overview of what you should aim to achieve:

1. Familiarize Yourself with the Basic Concepts
Start by understanding the theory behind diffusion models, particularly focusing on the papers and resources provided
Explore the U-Net architecture, which is pivotal in many diffusion model implementations
2. Implement a Basic Diffusion Model
Using the provided code and references, implement a basic version of a diffusion model
Train this model using the MNIST dataset to understand the nuances of training such models
3. Develop a Conditional Diffusion Model
Enhance your basic model to make it conditional, allowing it to generate specific digits from the MNIST dataset
This involves modifying the model to take a digit class as input and generate an image of that digit
4. Experiment and Iterate
Experiment with different parameters and architectures to improve the quality of the generated images
Document your findings and the impact of various changes on the model's performance
5. Theera Deployment
Deploy your model in a way that it can receive a digit class (0-9) as input and return a generated image of that digit
Learning outcomes
By the end of the week, you should have a solid understanding of:

Theoretical Foundations of Diffusion Models
U-Net Architecture and its Role in Image Synthesis
Training and Tuning of Diffusion Models
Conditional Generation using Diffusion Models
Code
In this phase, we encourage you to use existing implementations to gain practical insights. This repository provides a direct example of a Conditional Diffusion Model applied to the MNIST dataset.

https://github.com/TeaPearce/Conditional_Diffusion_MNIST

Study the Code
Examine the code in the provided repository. Understand how the different components of the model are implemented and how they interact with each other.

Link Theory with Practice
As you explore the code, continually refer back to the theoretical concepts in the papers about diffusion models and the U-Net architecture. This practice will help you understand why certain approaches are taken in the code and how they relate to the underlying principles of diffusion models.

Don't Just Copy-Paste
Avoid simple copy and paste. The goal is to understand the mechanisms behind the model.

Experiment and Modify
Once you have a grasp of both the theoretical and practical aspects, experiment with the code. Try modifying different aspects and observe how these changes impact the model's performance. Or experiment with CIFAR-10 dataset or Tiny ImageNet.

References
U-Net: Convolutional Networks for Biomedical Image Segmentation

Paper: https://arxiv.org/pdf/1505.04597.pdf
Deep Unsupervised Learning using Nonequilibrium Thermodynamics

Paper: https://arxiv.org/pdf/1503.03585.pdf
Denoising Diffusion Probabilistic Models

Paper: https://arxiv.org/pdf/2006.11239.pdf
High-Resolution Image Synthesis with Latent Diffusion Models

Paper: https://arxiv.org/abs/2112.10752