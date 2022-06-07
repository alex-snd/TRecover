---
title: "💪 Train"
---

# 💪 Train

To quickly start training the model, use this 
[Jupyter Notebook](https://colab.research.google.com/github/alex-snd/TRecover/blob/master/notebooks/TRecover-train-alone.ipynb).

=== "💻 Local"
    After the dataset is loaded, you can start training the model:
    ```
    trecover train local \
    --project-name {project_name} \
    --exp-mark {exp_mark} \
    --train-dataset-size {train_dataset_size} \
    --val-dataset-size {val_dataset_size} \
    --vis-dataset-size {vis_dataset_size} \
    --test-dataset-size {test_dataset_size} \
    --batch-size {batch_size} \
    --n-workers {n_workers} \
    --min-noise {min_noise} \
    --max-noise {max_noise} \
    --lr {lr} \
    --n-epochs {n_epochs} \
    --epoch-seek {epoch_seek} \
    --accumulation-step {accumulation_step} \
    --penalty-coefficient {penalty_coefficient} \
    
    --pe-max-len {pe_max_len} \
    --n-layers {n_layers} \
    --d-model {d_model} \
    --n-heads {n_heads} \
    --d-ff {d_ff} \
    --dropout {dropout}
    ```
    For more information use `trecover train local --help`

=== "🕸️ Distributed"
    :soon: TODO