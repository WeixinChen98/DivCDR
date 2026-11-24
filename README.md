# [WSDM 2027] Beyond Knowledge Transfer: How User Overlap Shapes Diversity in Cross-Domain Recommendation

The current code implements **DivCDR v1** from our original SIGIR submission. We revised the method in response to reviewer feedback and will release the implementation of the WSDM 2027 version once it is ready.

## Requirements

Before running the code, make sure the following dependencies are installed:

```
recbole==1.0.1
torch>=1.7.0
python>=3.7.0
```

## Quick-Start Guide

To begin using the DivCDR library, use the following command:

```bash
python run_recbole_cdr.py --model=[model] --config_files=[config_file]
```

### Example: Running DivCDR on the TenRec QB Dataset

```bash
python run_recbole_cdr.py --model=DivCDR --config_files=./recbole_cdr/properties/dataset/TenRec_QB.yaml
```

## Configuration and Model Selection

The configuration files define the dataset and hyperparameter settings. Example configurations can be found in the `./recbole_cdr/properties/dataset/` directory.  

### Example Config File: `TenRec_QB.yaml`

```yaml
field_separator: ","
source_domain:
  dataset: TenRec_QBVideo
  data_path: './recbole_cdr/dataset'
  USER_ID_FIELD: user_id
  ITEM_ID_FIELD: item_id
  RATING_FIELD: rating
  TIME_FIELD: timestamp
  NEG_PREFIX: neg_
  LABEL_FIELD: label
  load_col:
    inter: [user_id, item_id, rating]
  user_inter_num_interval: "[1,inf)"
  item_inter_num_interval: "[1,inf)"
  val_interval:
    rating: "[3,inf)"
  drop_filter_field: True

target_domain:
  dataset: TenRec_QBArticle
  data_path: './recbole_cdr/dataset'
  USER_ID_FIELD: user_id
  ITEM_ID_FIELD: item_id
  RATING_FIELD: rating
  TIME_FIELD: timestamp
  NEG_PREFIX: neg_
  LABEL_FIELD: label
  load_col:
    inter: [user_id, item_id, rating]
  user_inter_num_interval: "[1,inf)"
  item_inter_num_interval: "[1,inf)"
  val_interval:
    rating: "[3,inf)"
  drop_filter_field: True

train_epochs: ["BOTH:300"]  
train_batch_size: 4096
eval_batch_size: 409600000
metrics: ["NDCG", "Hit", "ShannonEntropy", "GiniIndex"]
topk: [5, 10]
valid_metric: NDCG@10
```

To customize experiments, modify the configuration file or provide additional parameters via command-line arguments.


## Acknowledgement

The implementation is based on the open-source recommendation library [RecBole-CDR](https://github.com/RUCAIBox/RecBole-CDR).

## Citation

If you find this repository helpful in your research, please consider citing our paper.

```bibtex
@inproceedings{zhao2027divcdr,
  author    = {Zhao, Yuhan and Chen, Weixin and Chen, Shu and Chen, Li},
  title     = {Beyond Knowledge Transfer: How User Overlap Shapes Diversity in Cross-Domain Recommendation},
  booktitle = {Findings of the Twentieth ACM International Conference on Web Search and Data Mining},
  series    = {WSDM '27},
  year      = {2027}
}
```
