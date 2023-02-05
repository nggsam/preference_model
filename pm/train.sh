# Sample commands to train
python train.py --pretrained_model=gpt2 --tokenizer_type=gpt2 --max_length=550 \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=64 \
--output_dir=.

