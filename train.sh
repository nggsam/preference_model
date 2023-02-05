# Sample commands to train
python train.py --pretrained_model=gpt2 --tokenizer_type=gpt2 --max_length=550 --eval_fraction=0.05 \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=64 \
--eval_steps=100 \
--save_steps=500 \
--warmup_steps=100 \
--learning_rate=1e-5 \
--output_dir=output \

