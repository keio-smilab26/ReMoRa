

model_path="checkpoints/ReMoRa-7B"
model_base="lmms-lab/LLaVA-Video-7B-Qwen2"
model_name="llava_qwen_lora"

results_dir=results/ReMoRa-7B

dataset_name=LongVideoBench
python llava/eval/infer.py \
    --model_path $model_path \
    --model_base $model_base \
    --model_name $model_name \
    --results_dir ${results_dir}/${dataset_name}_val \
    --max_frames_num 64 \
    --dataset_name $dataset_name \
    --data_path DATAS/eval/LongVideoBench/formatted_dataset.json \
    --video_root "path_to_video_folder" \
    --cals_acc
