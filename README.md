# R3 : Refined Retriever-Reader pipeline for Multidoc2dial

This is the implementation of our (team CMU_QA) submission to the [DialDoc 2023](https://doc2dial.github.io/workshop2022/) shared task (ranked 1st in the unseen-setting). Please check our [paper](https://aclanthology.org/2022.dialdoc-1.17/) for more details.


```
git clone https://github.com/srijan-bansal/R3_Mulltidoc2dial.git
cd R3_Mulltidoc2dial
conda activate multidoc2dial/conda_env.yml
```

---


### Download & Preprocess Multidoc2dial Dataset

```
export HF_HOME=cache
bash preprocess.sh
```

## Retriever 
These scripts are based on [splade](https://github.com/naver/splade/tree/main) retriever.

1. setup training data (use DPR retrieved outputs as hard negatives for retrieval traning)

```
bash retriever/scripts/prepare_retrieval_data.sh
```

<!-- 2. setup checkpoint (IGNORE)

download weights from [here](https://github.com/naver/splade/tree/main/weights) and paste the weights directory to ```retriever/splade_weights``` -->

<!-- #bash train.sh  # Fine tuning -->

2. Training (DPR negatives)
```
bash retriever/scripts/train_dpr_negatives.sh # 
```

3. Inference & Evaluation
```
bash retriever/scripts/eval.sh 

```
---

 
### Reranker
```
python
```

### Reader
```
bash reader/train_reader.sh temp/FID_train_10_top1_v1.json temp/FID_val_25.json top_102 t5-base checkpoint

```



