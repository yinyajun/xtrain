lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen1.5-7B-Chat,dtype=bfloat16 \
  --tasks ifeval,gsm8k,cmmlu \
  --device cuda:0 \
  --batch_size auto \
  --output_path ./eval/quick
