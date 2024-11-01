## TODOs
- Add vllm to our dependency


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


## Slurm commands
srun --gres=gpu:4 --time=8:00:00 --exclude=compute-permanent-node-365 --pty /bin/bash


## Training

### Test Runs
ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes=1 --main_process_port=1234 --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_sft.py recipes/zephyr-7b-beta/sft/config_lora.yaml


#### SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes=4 --main_process_port=1234 scripts/run_sft.py recipes/zephyr-7b-beta/sft/config_full.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes=4 --main_process_port=1234 scripts/run_sft.py recipes/zephyr-7b-gemma/sft/config_full.yaml

sbatch --job-name=gemma-2b-sft --nodes=1 --time=24:00:00 --gpus-per-node=4 recipes/launch.slurm zephyr-7b-gemma sft full deepspeed_zero3 '--model_name_or_path=google/gemma-7b --num_train_epochs=1 --output_dir=save/sft-gemma-7b'


#### RM
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes=4 --main_process_port=1234 scripts/run_reward_modeling.py recipes/zephyr-7b-beta/reward_modeling/config_full.yaml

sbatch --job-name=llama-2-7b-ultrachat200k-2e-rm --nodes=1 --time=8:00:00 --gpus-per-node=4 recipes/launch.slurm zephyr-7b-beta reward_modeling full deepspeed_zero3 '--model_name_or_path=kykim0/Llama-2-7b-ultrachat200k-2e --max_length=1024 --gradient_accumulation_steps=8 --learning_rate=2.0e-05 --output_dir=save/llama2-7b-rm-t15-f --run_name-llama2-7b-rm'

sbatch --job-name=g7b-sft-btb0.0 --nodes=1 --time=16:00:00 --gpus-per-node=4 recipes/launch.slurm zephyr-7b-gemma rm full deepspeed_zero3 '--model_name_or_path=kykim0/gemma-7b-ultrachat-sft --output_dir=save/g7b-uf-capo-rm --num_train_epochs=3 --bt_beta=0.0'



ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes=4 --main_process_port=1234 scripts/run_rm.py recipes/zephyr-7b-gemma/rm/config_full_7b.yaml --learning_rate=3.0e-07 --bt_beta=0.5 --seed=42




/data/kyuyoung_kim/dev/alignment-handbook/save/sft-llama3-8b-itt

/data/kyuyoung_kim/dev/alignment-handbook
save/l38b-itt-uf-capo-rm/l38b-itt-b32-lr1e-06-s0-e2-btbinf-seed42/checkpoint-1797
save/l38b-itt-uf-capo-rm/l38b-itt-b32-lr1e-06-s0-e2-btb0.5-seed42/checkpoint-3594

for n in 16 64 256; do CUDA_VISIBLE_DEVICES=0 python3 score_model_answer.py --model-path /data/kyuyoung_kim/dev/alignment-handbook/save/l38b-itt-uf-capo-rm/l38b-itt-b32-lr1e-06-s0-e2-btbinf-seed42/checkpoint-1797 --model-id llama-3 --num-gpus-total 1 --answer-file data/mt_bench/model_answer/l38b-sft-itt-512.jsonl --bon-file data/mt_bench/model_answer/l38b-sft-itt-$n-b32-btbinf-1e-s42.jsonl --best-of-n $n; done

for n in 16 64 256; do CUDA_VISIBLE_DEVICES=0 python3 score_model_answer.py --model-path /data/kyuyoung_kim/dev/alignment-handbook/save/l38b-itt-uf-capo-rm/l38b-itt-b32-lr1e-06-s0-e2-btb0.5-seed42/checkpoint-3594 --model-id llama-3 --num-gpus-total 1 --answer-file data/mt_bench/model_answer/l38b-sft-itt-512.jsonl --bon-file data/mt_bench/model_answer/l38b-sft-itt-$n-b32-btb0.5-2e-s42.jsonl --best-of-n $n; done



#### PPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port=1234 scripts/run_ppo.py recipes/zephyr-7b-beta/ppo/config_full.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port=1234 scripts/run_ppo.py recipes/zephyr-7b-beta/ppo/config_full.yaml --learning_rate=3e-5 --ppo_epochs=2 --init_kl_coef=0.05 --adap_kl_ctrl=true --target=6 --batch_size=32 --gradient_accumulation_steps=32 --eval_freq=50


sbatch --job-name=gemma-2b-rm-ppo-n1 --nodes=1 --time=24:00:00 --gpus-per-node=4 recipes/launch.slurm zephyr-7b-beta ppo full multi_gpu '--batch_size=16 --gradient_accumulation_steps=16 --reward_model=kykim0/Llama2-2e-data-gemma-2b-rm-1e --output_dir=save/gemma-2b-ours-rm-ppo'

sbatch --job-name=gemma-2b-uf-rm-ppo-n1 --nodes=1 --time=24:00:00 --gpus-per-node=4 recipes/launch.slurm zephyr-7b-beta ppo full multi_gpu '--batch_size=16 --gradient_accumulation_steps=16 --reward_model=kykim0/ultrafeedback-gemma-2b-rm-1e --output_dir=save/gemma-2b-uf-rm-ppo --init_kl_coef=0.1'


sbatch --job-name=l7b-t1.5g2brm-ppo --nodes=1 --time=48:00:00 --gpus-per-node=4 recipes/launch.slurm zephyr-7b-beta ppo full multi_gpu '--learning_rate=3e-5 --ppo_epochs=2 --init_kl_coef=0.05 --adap_kl_ctrl=true --target=6 --batch_size=32 --gradient_accumulation_steps=32 --eval_freq=50 --output_dir=gemma-2b-t1.5-ufall-rm-ppo-min384 --reward_model=kykim0/Llama2-2e-t1.5-ufall-gemma-2b-rm-1e --output_min_length=384'


#### MT-Bench Eval
python3 gen_model_answer.py --model-path kykim0/gemma-2b-ultrachat-sft --model-id gemma-custom --max-new-token 512
python3 gen_judgment.py --model-list gemma-custom --parallel 2
python3 show_result.py --input-file data/mt_bench/model_judgment/gpt-4_single_g7b.jsonl


### Launch on Slurm and override default hyperparameters
sbatch --job-name=llama2-sft-2epoch --nodes=1 recipes/launch2.slurm zephyr-7b-beta sft full deepspeed_zero3 '--output_dir=data/Llama-2-7b-hf-sft-full-2epoch --num_train_epochs=2'

## Evaluation
python3 eval/eval_mtbench.py --response-file-path ~/dev/FastChat/fastchat/llm_judge/data/mt_bench/model_answer/Llama-2-7b-hf.jsonl --question-file-path ~/dev/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl --model-path meta-llama/Llama-2-7b-hf --output-path ./eval_results --metrics fkgl num_char ppl --device-map cuda:0 --batch-size 2 --dataset mt_bench

            # for idx, (q, r) in enumerate(zip(batch["query"], batch["response"])):
            #     print(f'({idx}):\n  Q ({len(query_tensors[idx])}): {q}\n  A ({len(response_tensors[idx])}): {r}')

self.accelerator.distributed_type

### TODOs
negative default score for not finished responses


python3 tests/reward_modeling.py --model_name_or_path=facebook/opt-350m --output_dir="reward_modeling_anthropic_hh" --per_device_train_batch_size=64 --num_train_epochs=1 --gradient_accumulation_steps=16 --gradient_checkpointing=True --learning_rate=1.41e-5 --report_to="wandb" --remove_unused_columns=False --optim="adamw_torch" --logging_steps=10 --evaluation_strategy="steps" --max_length=512


# need to set mixed_precision: 'no'
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes=4 --main_process_port=1234 tests/reward_modeling.py --model_name_or_path=kykim0/Llama-2-7b-ultrachat200k-2e --output_dir="reward_modeling_anthropic_hh" --per_device_train_batch_size=1 --num_train_epochs=1 --gradient_accumulation_steps=16 --gradient_checkpointing=True --learning_rate=1.41e-5 --report_to="wandb" --remove_unused_columns=False --optim="adamw_torch" --logging_steps=10 --evaluation_strategy="steps" --max_length=512

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes=4 --main_process_port=1234 tests/reward_modeling.py --model_name_or_path=facebook/opt-350m --output_dir="reward_modeling_anthropic_hh" --per_device_train_batch_size=1 --num_train_epochs=1 --gradient_accumulation_steps=16 --gradient_checkpointing=True --learning_rate=1.41e-5 --report_to="wandb" --remove_unused_columns=False --optim="adamw_torch" --logging_steps=10 --evaluation_strategy="steps" --max_length=512

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes=4 --main_process_port=1234 --mixed_precision=no tests/reward_modeling.py --model_name_or_path=google/gemma-7b --output_dir="reward_modeling_anthropic_hh" --per_device_train_batch_size=1 --num_train_epochs=1 --gradient_accumulation_steps=8 --gradient_checkpointing=True --learning_rate=1.41e-5 --report_to="wandb" --remove_unused_columns=False --optim="adamw_torch" --logging_steps=10 --evaluation_strategy="steps" --max_length=512 --run_name=gemma-ours-bs32-ds3 --bf16
