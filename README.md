# RxRx19-analysis

This is a brief exploratory analysis of the [RxRx19 dataset released by Recursion Pharma.](https://www.rxrx.ai/rxrx19a) I've been meaning to dig into this for months, and I am already impressed!

In summary, Recursion has treated cells with drugs, and then infected them with COVID19. Afterwards, high-resolution microscopic imagery is performed on multiple flourescent channels that target distinct cellular structures. Following this, a deep learning model condenses this large image dataset into embedded vectors of 1024 dimensions that describe each image.

I have no idea how their deep learning model is trained, but it is effectively a black box that reduces the dimensionality of the image. The cellular images aren't completely representative of the underlying cell state, which itself is a black box. Somewhere in this stack is the real cellular state (s), hidden from us.

s -> [black box inner workings of the cell] -> [visual appearance] -> [recursions model] -> e

With these embedding vectors, we have a birds-eye view that only partially reveals the cells underlying state, but its still useful! Its also much cheaper than more direct measurements, such as single-cell sequencing to determine gene expression levels. 

For drug repurposing, it is useful to consider the idea that the underlying cell state is perturbed by that drug. Furthermore, insults such as viral infection will also perturb this state. Some of that perturbation will hopefully be projected through this stack of black boxes to be optically detected with the handful of dyes used! To find drugs for COVID19, what we need to define is some kind of baseline homeostatis of the cells. Later, if we find drugs that return infected cells to that homeostasis, they may be viable candidates.

## Preliminary exploration

