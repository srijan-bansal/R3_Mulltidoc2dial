# R3 : Refined Retriever-Reader pipeline for Multidoc2dial

This is the implementation of our (team CMU_QA) submission to the [DialDoc 2023](https://doc2dial.github.io/workshop2022/) shared task (ranked 1st in the unseen-setting). Please check our [paper](https://aclanthology.org/2022.dialdoc-1.17/) for more details.


<br />
clone this repository

```
cd R3_Mulltidoc2dial/multidoc2dial
conda activate conda_env.yml
export HF_HOME=../../cache
```

Download & Preprocess Multidoc2dial Dataset

```
cd scripts
bash run_download.sh
bash run_data_preprocessing.sh structure grounding
bash run_data_preprocessing.sh structure generation
bash run_data_preprocessing_dpr.sh structure all 
```

Retriever 
```

```

Reranker
```

```

Reader
```
bash reader/train_reader.sh temp/FID_train_10_top1_v1.json temp/FID_val_25.json top_102 t5-base checkpoint
```




