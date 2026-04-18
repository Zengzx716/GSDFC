# GSDFC
**"A Greedy Band Selection Strategy Based on Dual-frequency Collaborative Feature Fusion for Unsupervised Hyperspectral Band Selection"**

**Abstract**--Band selection plays a crucial role as a preprocessing step in hyperspectral image processing tasks. Currently, most band selection methods tend to fuse spatial spectral features, directly from the original spectral domain, without fully exploring the distinct information embedded in different frequency components of the spectrum. In this study, a greedy band selection strategy based on dual-frequency collaborative feature fusion (GSDFC) is proposed for unsupervised band selection of hyperspectral images. Firstly, a dual-frequency collaborative feature fusion module (DFCFFM) is designed to extract and fuse the spatial spectral features of two frequency domain components separately. Secondly, to further enhance the effectiveness of DFCFFM and learn important features in low-frequency components adaptively, a grouping spatial attention module (GSAM) is introduced, which effectively captures global and local spatial dependencies. Finally, a greedy selection strategy is constructed by utilizing the mutual information (MI) of the fused features and the Pearson correlation coefficient (PCC) of the original data to select the optimal subset of bands. The proposed GSDFC band selection method is validated by classification. Extensive experiments have shown that, compared with some advanced methods, the proposed GSDFC method can achieve the best classification performance on three public datasets: Indian Pines, Pavia University, and Houston 2013, which fully demonstrates the effectiveness of the proposed GSDFC method.

data_load.py----Upload hyperspectral image and perform preprocessing.

GBS.py----Corresponding to greedy band selection strategy.

DFCFFM.py, DWT_2D.py, DWT_3D_N.py and ema.py----Corresponding to The Implemental process of DFCFFM.

GSDFC.py----Corresponding to the main code.
