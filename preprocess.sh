cd multidoc2dial/scripts


bash multidoc2dial/scripts/run_download.sh


DATA_DIR=data/



task=grounding

task=generation




python data_preprocessor.py \
--dataset_config_name multidoc2dial \
--output_dir $DATA_DIR/mdd_all \
--kb_dir $DATA_DIR/mdd_kb \
--segmentation structure \
--task $task 





bash run_data_preprocessing.sh structure grounding
bash run_data_preprocessing.sh structure generation
bash run_data_preprocessing_dpr.sh structure all 