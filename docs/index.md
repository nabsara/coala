<div style="text-align: center"> 

<h1>CoALa: A Plugin model for Controlling Audio Latents</h1>
Supporting webpage for ISMIR 2026.
Code is available <a href="https://https://github.com/nabsara/coala">here</a>.
</div>

<br>

*This webpage will be updated soon with all the qualitative results.*

<div style="text-align: justify"> 

<h3>Abstract</h3>

In recent years, neural audio synthesis has advanced significantly, enabling high-quality real-time waveform generation on standard CPUs. However, defining personalized controls for live musical performance remains challenging, as existing methods typically require large labeled datasets and rely on heavy architectures that are generally impractical for real-time use. In this paper, we propose CoALa, a lightweight plugin model for Controlling Audio Latents in real-time, which enables users to add time-varying musical controls on pretrained neural audio synthesis models using only a limited set of labeled examples. Our method leverages a variational autoencoder with a contrastive learning regularization to disentangle control-relevant features from the pretrained latent space in designated dimensions. We rely on labeled frames as an augmentation strategy to generate multiple views of similar control variations for each unlabeled latent sequence, and apply the contrastive objective on these variation embeddings. By augmenting the pretrained latent space with temporal latent tokens, our method supports an arbitrary number of control variables, generalizes to highly compact latent representations, and can simultaneously disentangle multiple time-varying controls evolving at different temporal scales. Our model defines an invertible mapping between both latent and control trajectories and achieves high control accuracy while preserving synthesis quality for feature extraction, editing, and conditional synthesis tasks, using as little as 10\% labeled data. Its lightweight streamable architecture further enables fast CPU inference, making it well suited for real-time applications in Max/MSP environments.
</div>

<br>
<br>

<div>
<p align="center">
<img src="assets/imgs/ismir26_overview.png" style="width: 40%; height: auto;">
</p>
<p style="text-align: center"><b>Figure 1: </b>Approach overview.</p>
</div>

<br>
<br>


Here, we provide audio examples and additional results to complete the experiments section from the paper and illustrate the control abilities of our proposed method. We organise the results into 3 parts:
- [Disentangling one control](#disentangling-one-control)
- [Disentangling multiple controls](#disentangling-multiple-controls)
- [Real-time application in MaxMSP](#real-time-application-in-maxmsp)

# Disentangling one control


[TO COMPLETE]


# Disentangling multiple controls

[TO COMPLETE]


# Real-time application in Max/MSP

[TO COMPLETE]

