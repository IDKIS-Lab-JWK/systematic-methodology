$percentages = 100
$numWorkers = 4

foreach ($p in $percentages) {
    python Code\Models\Merlin\train_conv_lora_llrd.py `
      --stage all `
      --percentage $p `
      --parameter_source publication_derived `
      --batch_size 2 `
      --num_workers $numWorkers `
      --publication_override_reason "Batch size adjusted for available GPU memory" `
      --model_source refit_full_development `
      --amp `
      --resume
}
