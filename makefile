DATA_DIR=/local/scratch/ssd
TRAIN_DIR=/local/scratch/yaz21/tmp
run: alexnet_multi_gpu_train.py
	CUDA_VISIBLE_DEVICES=0,1 python alexnet_multi_gpu_train.py --load_from_checkpoint=True --num_preprocess_threads=4 --num_gpus=2 --train_dir=$(TRAIN_DIR) --data_dir=$(DATA_DIR)

git-add:
	git add -A
	git commit -m"alexnet first commit"
	git push

git-fetch:
	git fetch
	git merge
