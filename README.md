# RxRx19-analysis

This is a brief exploratory analysis of the [RxRx19 dataset released by Recursion Pharma.](https://www.rxrx.ai/rxrx19a) I've been meaning to dig into this data for months, and now I have an excuse!

In summary, Recursion has treated cells with drugs, and then infected them with COVID19. Afterwards, high-resolution microscopic imagery is performed on multiple flourescent channels that target distinct cellular structures. Following this, a deep learning model condenses this large image dataset into embedded vectors of 1024 dimensions that describe each image.

I have no idea how their deep learning model is trained, but it is effectively a black box that reduces the dimensionality of the image. The cellular images aren't completely representative of the underlying cell state, which itself is a black box. Somewhere in this stack is the real cellular state (s), hidden from us underneath multiple layers.

biological state -> [cellular black box] -> visual appearance -> [just 5 dyes] -> [recursions black box] -> embedding 

With these embedding vectors, we have a birds-eye view that only partially reveals the cells underlying state, but its still useful! Its also much cheaper than more direct measurements, such as single-cell sequencing to determine gene expression levels. 

For drug repurposing, it is useful to consider the idea that the underlying cell state is perturbed by that drug. Furthermore, insults such as viral infection will also perturb this state.

For drug repurposing, its useful to consider that both disease insults (like covid infection) and drug exposure will perturb the underlying cellular state, and hopefully shift the images enough to show a clear perturbation in the embedding vector. To find drugs for COVID19, what we need to define is some kind of baseline homeostatis of the cells. Later, if we find drugs that return infected cells to that homeostasis, they may be viable candidates.

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

I've been looking for more opportunities to play with tSNE embedding, and this problem lends itself to this technique. I start with plots on 5% of the full shuffled dataset. The most important, and enthusiastic finding, is that the embedding vectors for the Mock and UV-inactivated cells tend to form significant clusters, suggesting that this homeostasis appears to be relatively stable from the perspective of the microscopy images.

|              |   |
:-------------------------:|:-------------------------:
![](results/no-normalization/tsne-disease_condition.png) | ![](results/no-normalization/tsne-experiment.png) 

The embedding vectors also form distinct clusters that correspond to both cell-type, and the experiment performed. By far most of the data are from the two HRCE experiments, each of which has its own associated cluster that is disjoint from the rest of the embedding space. The tSNE embedding has revealed that there is a systematic bias between experimental replicates.

What could have caused this bias? Many things! Experiments could have been run on different days, leading to differences in microscope aquisition settings, a slightly different concentration of one of the channel dies, or any other factors of prepping the samples. Regardless of the cause, the images from the experiment were ever-so-slightly different, and recursions deep learning model translated these differences into euclidean transformations in the embedding vector space. TSNE just made this shift easier to see.

This transformation can be easily corrected however! We can just compute centroid vectors for each of the 4 experiments, and then subtract this centroid from each vector in turn. Doing so centers the dataset. These vectors could also be normalized by their respective standard deviations.  Running the tSNE again reveals that the clusters for the same cell type collapse!  

|              |   |
:-------------------------:|:-------------------------:
![](results/experiment-norm/tsne-disease_condition.png) | ![](results/experiment-norm/tsne-experiment.png) 


### exploring other biases

The following are results from an overnight tSNE run on the **entire** dataset, (but without the previous normalization). With a larger number of points, it is possible to see other categorical variables that are somehow distinguishable by the neural network. The plate number for example shows quite clearly that all Vero samples were collected on just a few plates. More intriguingly, the site imaged within each well shows an obvious "clumpiness", which implies that **the deep learning model is sometimes able to distinguish which of the 4 corners of a well it is imaging.** I'll need to look at the image dataset later to learn why! I wonder if it can spot the walls!

Importantly, these other confounds are just as easily normalized as the experimental run. For each plate or site, centroids can be subtracted. It could also be argued that the four sites of each well could have their embedding vectors averaged, but such a strategy is probably best assessed in a larger pipeline.

|              |   |
:-------------------------:|:-------------------------:
![](results/first-10percent/tsne-site.png)  |  ![](results/first-10percent/tsne-plate.png)
![](results/first-10percent/tsne-concentration.png)  |  ![](results/first-10percent/)

I mapped concentration too. Indeed a few clusters appear, specifically of higher-concentrations of drugs. I can imagine a few distinct causes of this. Large detoxification activity in the cells perhaps? Or even a direct photometric effect of the drug on the flourescence signals of a specific channel?


### Post-correction .... how can we find leads?

Taking a cue from the "connectivity map" project, and other similar drug repurposing efforts based on gene expression perturbations, drugs for a given illness could be found if they perturb a cellular state in a direction opposite that of the illness. In the embedding space, one could imagine finding vectors that are antipodal to the disease perturbations.

With this dataset however, what is missing is data on the cellular states induced by each drug applied to uninfected cells! While sufficient healthy control cells are present to obtain a nice homeostatic baseline, the rest of the data is some mixture of that drugs perturbation and the perturbation induced by the virus. This can confound our efforts.

Its a fair assumption that for most drugs, the lowest concentrations have small effects on the cell. Since there is little "clumping" of the concentrations in the tSNE embedding, I am inclined to believe that the effect of the coronavirus itself already has significant effects on the variance of the embedding vectors. Indeed, the "area" of this tSNE embedding showing infected cells is quite large.

Thus, it seems that a reasonable starting point for finding lead compounds is to look within the "neighborhood of homeostasis" in this unsupervised embedding. This is breaking a few rules! Indeed, euclidean distances are only reliable over short distances in such embeddings! Comparisons should also be performed for straight kNN in an embedding provided by PCA, or the full embedding dimension itself.



### A full overnight run, with all variables centered

Running tSNE on the full dataset takes many hours, so I committed to centering 4 of the categorical variables in the dataset. : ['experiment', 'site', 'plate', 'disease_condition']

Most relevantly, it became clear that centering on "disease condition" is likely required, because of how I assume the plates were prepared. Cells were most likely infected in batch, and then spread to each well. This batch may or may not have been prepared independently of each plate. 

Centering the embedding vectors results in multiple homeostatic clusters featured prominently in the center of the manifold. I find this encouraging, as this suggests that the cells can take on a number of normal states. Also encouraging is that the Mock and UV-inactivated preparations produce the same 3 overlapping clusters.

|              |   |
:-------------------------:|:-------------------------:
![](results/normalization-overnight/tsne-disease_condition.png)  |  ![](results/normalization-overnight/tsne-experiment.png)
![](results/normalization-overnight/tsne-treatment_conc.png)  |  ![](results/normalization-overnight/tsne-plate.png)


### Better approaches

One could (ab)use a supervised training method as well. Consider a classifier which maps the 1024-dimensional embedding vector to a sigmoidal output that predicts uninfected/infected. This would need to be carefully regularized so as to not overtrain. 

Once trained however, drugs could be ranked by finding embedding vectors that produce the highest output probabilities of an "uninfected" classification.


### Crazy ideas

Splitting a well into 4 sites is a clever way of creating replicates, but this approach can be generalized to smaller and smaller divisions of each well. Taking this to the extreme, one could segment the images into individual cells, and train a model to produce embedding vectors for each! This strategy might better pick up morphological changes that only impact small numbers of cells in the well.

I suspect the best way to train such a model would be with a triplet loss function! Borrowed from facial recognition, this technique could produce vectors that describe the "identity" of cells in a specific treatment state, but invariant to positions and orientation. When triplet loss is applied to face recognition, identity vectors are produced that uniquely identify a face regardless of the specific scene that face is found it.

### Fun ideas

I want to find a clever model that incorporates the SMILES strings alongside the embedding vectors. Will be thinking of this!





# Install notes

install cudatsne for faster tSNE computations on the GPU

this took way too much time ..... using ubuntu 20 might be easier ....

	* compile and install cmake : https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line
	* install cuda-10-2
	* install intel MKL
	* clone faiss repo, compile, install : https://github.com/facebookresearch/faiss/blob/master/INSTALL.md
		* need to manually edit makefiles to point to intel MKL, python libs, etc ..... 
	* pip3.6 install faiss-gpu
	* pip3.6 install tsnecuda 


