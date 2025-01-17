<img src="src/gluer/data/gluer_icon.png" align="left" width="180" height="180"> <h1 style=“text-align:left”>GLUER: integrative analysis of multi-omics and imaging data at single-cell resolution by deep neural networks </h1>
<br>

## GLUER has been implemented in Python.

Single-cell omics assays such as RNA-Seq, ATAC-Seq and methylome sequencing have been developed to identify cell types and/or states in heterogeneous tissues. However, it is challenging to integrate these different types of single-cell omics data. Moreover, the advent of spatially resolved single-cell imaging data represent another challenge for integrative analysis with omics data. Here, we present a new algorithm, inteGrative anaLysis of mUlti-omics at single-cEll Resolution (GLUER), for integration of single-cell multi-omics data as well as imaging data. We tested GLUER using multiple datasets generated using multiomics data generated on the same single cells, which was taken as the ground truth. Our results demonstrate that GLUER has significantly improved performance in terms of the accuracy of matching cells with different data modalities, which in turn enhances downstream analyses such as clustering and trajectory inference. GLUER provides a principled analytical framework for studying the heterogeneity of cell populations using multi-omics and imaging data.

<p align="center">
<img src="src/gluer/data/overview.png">
</p>

### Install from source codes
#### standalone installation
In Command Window type:
```
git clone https://github.com/software-github/GLUER/tree/main
cd GLUER0/dist
pip install GLUER.tar.gz
or
pip install GLUER.whl
```


### Quick start
```
import numpy as np
import pandas as pd
import scanpy as sc
import gluer as gr

# load the data
rna_data, acc_data, gluer_data = gr.load_demo_data()

# run GLUER
gluer_obj = gr.gluer(rna_data, acc_data, batch_categories=['RNA','ACC'])

# run umap_cell_embeddings
gluer_obj = gr.run_umap(gluer_obj,n_neighbors = 40, min_dist = 0)

# Visualize the integration results in UMAP
sc.pl.embedding(gluer_obj,
                'umap_cell_embeddings',
                color=['gluer_batch'],
                s=20,
                title='GLUER DEMO',
                show=False)
```                   
### Explore the data in Dashboard

This dashboard is to explore the integrative analysis results using GLUER. The major functions:
* show the co-embeded data, the reference dataset, and the query datasets in the same window
* the three windows are idependent from each other.
* the differential analysis and downloading funciton of plotting data will make users to custermize their analysis needs.

More details of this Dashboard are found here. We have been keeping updating the functions of this dashboard. Please let me know if you have any suggestions for this dashboard.

<p align="center">
<img src="src/gluer/data/GUI_tutorial.png">
</p>



## Help
Please feel free to contact Tao Peng (software.github@gmail.com) or Kai Tan (tank1@chop.edu) if you have any questions about the software.

## Reference
Peng, Tao, et al. "GLUER: integrative analysis of multi-omics and imaging data at single-cell resolution by deep neural networks"
