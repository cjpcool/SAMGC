#  SAMGC

This is the code of paper: Shared-Attribute Multi-graph Clustering with Global Self-Attention


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

 



