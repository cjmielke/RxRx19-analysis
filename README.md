# RxRx19-analysis

This is a brief exploratory analysis of the [RxRx19 dataset released by Recursion Pharma.](https://www.rxrx.ai/rxrx19a) I've been meaning to dig into this for months, and I am already impressed!

In summary, Recursion has treated cells with drugs, and then infected them with COVID19. Afterwards, high-resolution microscopic imagery is performed on multiple flourescent channels that target distinct cellular structures. Following this, a deep learning model condenses this large image dataset into embedded vectors of 1024 dimensions that describe each image.

I have no idea how their deep learning model is trained, but it is effectively a black box that reduces the dimensionality of the image. The cellular images aren't completely representative of the underlying cell state, which itself is a black box. Somewhere in this stack is the real cellular state (s), hidden from us.

s -> [black box inner workings of the cell] -> [visual appearance] -> [recursions model] -> e

With these embedding vectors, we have a birds-eye view that only partially reveals the cells underlying state, but its still useful! Its also much cheaper than more direct measurements, such as single-cell sequencing to determine gene expression levels. 

For drug repurposing, it is useful to consider the idea that the underlying cell state is perturbed by that drug. Furthermore, insults such as viral infection will also perturb this state. Some of that perturbation will hopefully be projected through this stack of black boxes to be optically detected with the handful of dyes used! To find drugs for COVID19, what we need to define is some kind of baseline homeostatis of the cells. Later, if we find drugs that return infected cells to that homeostasis, they may be viable candidates.

## Preliminary exploration

First, a breakdown of the condition of each well shows a justifiably unbalanced dataset. Still, quite a lot of controls!

	Active SARS-CoV-2            280376
	Mock                           9120
	UV Inactivated SARS-CoV-2      9120


There are two cell types, split into 4 distinct experiments

	========== cell_type === unique values :  2
	HRCE    284080
	VERO     21440
	========== experiment === unique values :  4
	HRCE-2    144720
	HRCE-1    139360
	VERO-1     10720
	VERO-2     10720


## Visualization

Ive been looking for an excuse to play with tSNE embedding! For this dataset, it seems to have worked wonders! I start with plots on 10% of the full shuffled dataset. The most important, and enthusiastic finding, is that the embedding vectors for the Mock and UV-inactivated cells tend to form significant clusters, suggesting that this homeostasis appears to be relatively stable from the perspective of the microscopy images.


![](results/first-10percent/tsne-disease_condition.png)
Despite this, the embedding vectors form distinct clusters that correspond to both cell-type, and the experiment performed. By far most of the data are from the two HRCE experiments, each of which has its own associated cluster. Within each of these experiments is a local cluster of cells in homeostasis. This is important, because the tSNE embedding has revealed that there is a systematic bias in those image datasets that propagated through the nearal network and created a euclidean separation of the corresponding embedding vectors. This isn't really a problem though! We can instead focus on single experiments for downstream analysis.

![](results/first-10percent/tsne-experiment.png)

Although there is significant bias between the 4 different experiments, the other categorical variables within the dataset are less concerning. The plate number shows some local clumping of data, as does the site imaged within each well. I'll need to look at the images later to find systematic reasons that these images are "distinguishable". 

|              |   |
:-------------------------:|:-------------------------:
![](results/first-10percent/tsne-site.png)  |  ![](results/first-10percent/tsne-plate.png)
![](results/first-10percent/)  |  ![](results/first-10percent/)


### Focusing on one experiment

Most relevantly, the fact that diseased vs non-diseased cells are so easily seperable by the embedding vectors is quite useful! Narrowing down to a single experiments datapoints is a next logical step. The following tSNE embeddings are computed (overnight) on ALL vectors from HRCE-1.

![](results/exp1-big/tsne-disease_condition.png)

