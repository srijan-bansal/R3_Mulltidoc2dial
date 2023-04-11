# R3_Mulltidoc2dial
This is the implementation for our paper R3 : Refined Retriever-Reader pipeline for Multidoc2dial

git clone https://github.com/srijan-bansal/R3_Mulltidoc2dial.git

cd R3_Mulltidoc2dial/multidoc2dial

# conda activate conda_env.yml


export HF_HOME=../../cache
bash run_download.sh

bash run_data_preprocessing.sh structure grounding
bash run_data_preprocessing.sh structure generation


bash run_data_preprocessing_dpr.sh structure all 





<!-- export `CHECKPOINTS` for -->



cd scripts


