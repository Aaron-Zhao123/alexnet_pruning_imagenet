DATA_DIR=/local/scratch/ssd
TRAIN_DIR=/local/scratch/yaz21/tmp
run: alexnet_multi_gpu_train.py
	python alexnet_multi_gpu_train.py --store_from_checkpoint=True --num_preprocess_threads=4 --num_gpus=4 --train_dir=$(TRAIN_DIR) --data_dir=$(DATA_DIR)

git-add:
	git add -A
	git commit -m"alexnet first commit"
	git push

git-fetch:
	git fetch
	git merge
