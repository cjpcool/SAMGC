#  SAMGC

This is the code of paper: Shared-Attribute Multi-graph Clustering with Global Self-Attention.

### Cite this paper
~~~
@InProceedings{10.1007/978-3-031-30105-6_5,
author="Chen, Jianpeng and Yang, Zhimeng and Pu, Jingyu and Ren, Yazhou and Pu, Xiaorong and Gao, Li and He, Lifang"
title="Shared-Attribute Multi-Graph Clustering with Global Self-Attention",
booktitle="Neural Information Processing",
year="2023",
publisher="Springer International Publishing",
address="Cham",
pages="51--63",
isbn="978-3-031-30105-6"
}
~~~
Chen, J. et al. (2023). Shared-Attribute Multi-Graph Clustering with Global Self-Attention. In: Tanveer, M., Agarwal, S., Ozawa, S., Ekbal, A., Jatowt, A. (eds) Neural Information Processing. ICONIP 2022. Lecture Notes in Computer Science, vol 13623. Springer, Cham. https://doi.org/10.1007/978-3-031-30105-6_5


# Requirements

* Python 3.7.11

* PyTorch 1.9.1

* munkres 1.1.4

* scikit-learn 0.24.2

* scipy 1.6.2


# Datasets

* ACM and Cora are included in `.\data\cora` and `.\data\` respectively.
* Large datasets will uploaded after review.

# Test SAMGC

* Test SAMGC on ACM: `sh test_acm.sh`
* Test SAMGC on Cora: `sh test_cora.sh`

# Train SAMGC

* Train SAMGC on ACM: `sh train_acm.sh`

* Train SAMGC on Cora: `sh train_cora.sh`

# Results of SAMGC

|          | NMI  | ARI  | ACC  | F1   |
| :------- | ---- | ---- | ---- | ---- |
| **ACM**  | 77.2 | 82.8 | 94.0 | 94.0 |
| **Cora** | 58.2 | 51.1 | 73.5 | 72.7 |






