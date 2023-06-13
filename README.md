Current OOM error when running on dual RTX 6000 Ada. Use `max_memory` arg? (I thought that was only for inference...)


```
% PYTHONPATH=. accelerate launch \
        --config_file config/accelerate_config.yaml \
        training/trainer.py \
        --input-model EleutherAI/pythia-12b \
        --epochs 2 \
        --output-dir output \
        --per-device-train-batch-size 2 \
        --per-device-eval-batch-size 2 \
        --logging-steps 10 \
        --save-steps 200 \
        --save-total-limit 20 \
        --eval-steps 50 \
        --warmup-steps 50 \
        --test-size-ratio 0.1 \
        --lr 5e-6 \
        --bf16 True
[2023-06-14 00:07:25,527] [INFO] [real_accelerator.py:110:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-06-14 00:07:27,285] [INFO] [real_accelerator.py:110:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2023-06-14 00:07:27,322] [INFO] [real_accelerator.py:110:get_accelerator] Setting ds_accelerator to cuda (auto detect)

===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please run

python -m bitsandbytes

 and submit this information together with your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
bin /home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/libbitsandbytes_cuda118.so
/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: /home/opyate/anaconda3/envs/opyate-dolly-py39 did not contain ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] as expected! Searching further paths...
  warn(msg)
/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('@/tmp/.ICE-unix/3441,unix/nil'), PosixPath('local/nil')}
  warn(msg)
/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/org/gnome/Terminal/screen/a66af974_82d0_4686_a40e_384e7121b93f')}
  warn(msg)
/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/usr/share/gconf/ubuntu.default.path')}
  warn(msg)
/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('0'), PosixPath('1')}
  warn(msg)
/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/usr/share/gconf/ubuntu.mandatory.path')}
  warn(msg)
/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/etc/xdg/xdg-ubuntu')}
  warn(msg)
/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/tmp/torchelastic_cswtagb6/none_8rttegyx/attempt_0/0/error.json')}
  warn(msg)
CUDA_SETUP: WARNING! libcudart.so not found in any environmental path. Searching in backup paths...
/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: Found duplicate ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] files: {PosixPath('/usr/local/cuda/lib64/libcudart.so.11.0'), PosixPath('/usr/local/cuda/lib64/libcudart.so')}.. We'll flip a coin and try one of these, in order to fail forward.
Either way, this might cause trouble in the future:
If you get `CUDA error: invalid device function` errors, the above might be the cause and the solution is to make sure only one ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] in the paths that we search based on your env.
  warn(msg)
CUDA SETUP: CUDA runtime path found: /usr/local/cuda/lib64/libcudart.so.11.0
CUDA SETUP: Highest compute capability among GPUs detected: 8.9
CUDA SETUP: Detected CUDA version 118
CUDA SETUP: Loading binary /home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/libbitsandbytes_cuda118.so...

===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please run

python -m bitsandbytes

 and submit this information together with your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
bin /home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/libbitsandbytes_cuda118.so
/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: /home/opyate/anaconda3/envs/opyate-dolly-py39 did not contain ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] as expected! Searching further paths...
  warn(msg)
/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('@/tmp/.ICE-unix/3441,unix/nil'), PosixPath('local/nil')}
  warn(msg)
/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/org/gnome/Terminal/screen/a66af974_82d0_4686_a40e_384e7121b93f')}
  warn(msg)
/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/usr/share/gconf/ubuntu.default.path')}
  warn(msg)
/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('1'), PosixPath('0')}
  warn(msg)
/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/usr/share/gconf/ubuntu.mandatory.path')}
  warn(msg)
/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/etc/xdg/xdg-ubuntu')}
  warn(msg)
/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/tmp/torchelastic_cswtagb6/none_8rttegyx/attempt_0/1/error.json')}
  warn(msg)
CUDA_SETUP: WARNING! libcudart.so not found in any environmental path. Searching in backup paths...
/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: Found duplicate ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] files: {PosixPath('/usr/local/cuda/lib64/libcudart.so'), PosixPath('/usr/local/cuda/lib64/libcudart.so.11.0')}.. We'll flip a coin and try one of these, in order to fail forward.
Either way, this might cause trouble in the future:
If you get `CUDA error: invalid device function` errors, the above might be the cause and the solution is to make sure only one ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] in the paths that we search based on your env.
  warn(msg)
CUDA SETUP: CUDA runtime path found: /usr/local/cuda/lib64/libcudart.so
CUDA SETUP: Highest compute capability among GPUs detected: 8.9
CUDA SETUP: Detected CUDA version 118
CUDA SETUP: Loading binary /home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/bitsandbytes/libbitsandbytes_cuda118.so...
2023-06-14 00:07:28 INFO [torch.distributed.distributed_c10d] Added key: store_based_barrier_key:1 to store for rank: 0
2023-06-14 00:07:28 INFO [torch.distributed.distributed_c10d] Added key: store_based_barrier_key:1 to store for rank: 1
2023-06-14 00:07:28 INFO [torch.distributed.distributed_c10d] Rank 1: Completed store-based barrier for key:store_based_barrier_key:1 with 2 nodes.
2023-06-14 00:07:28 INFO [__main__] Loading tokenizer for EleutherAI/pythia-12b
2023-06-14 00:07:28 INFO [torch.distributed.distributed_c10d] Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 2 nodes.
2023-06-14 00:07:28 INFO [__main__] Loading tokenizer for EleutherAI/pythia-12b
2023-06-14 00:07:28 INFO [__main__] Loading model for EleutherAI/pythia-12b
2023-06-14 00:07:28 INFO [__main__] Loading model for EleutherAI/pythia-12b
2023-06-14 00:07:30 WARNING [accelerate.utils.modeling] The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function.
2023-06-14 00:07:30 WARNING [accelerate.utils.modeling] The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:21<00:00,  7.30s/it]
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:21<00:00,  7.33s/it]
2023-06-14 00:08:12 INFO [__main__] trainable params: 5898240 || all params: 11851970560 || trainable%: 0.05
2023-06-14 00:08:13 INFO [__main__] trainable params: 5898240 || all params: 11851970560 || trainable%: 0.05
2023-06-14 00:08:13 ERROR [__main__] main failed
Traceback (most recent call last):
  File "/home/opyate/Documents/code/github/dolly/training/trainer.py", line 423, in <module>
    main()
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/click/core.py", line 1130, in __call__
    return self.main(*args, **kwargs)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/click/core.py", line 1055, in main
    rv = self.invoke(ctx)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/click/core.py", line 1404, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/click/core.py", line 760, in invoke
    return __callback(*args, **kwargs)
  File "/home/opyate/Documents/code/github/dolly/training/trainer.py", line 395, in main
    train(**kwargs)
  File "/home/opyate/Documents/code/github/dolly/training/trainer.py", line 273, in train
    model, tokenizer = get_model_tokenizer(
  File "/home/opyate/Documents/code/github/dolly/training/trainer.py", line 209, in get_model_tokenizer
    model.resize_token_embeddings(len(tokenizer))
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/transformers/modeling_utils.py", line 1397, in resize_token_embeddings
    model_embeds = self._resize_token_embeddings(new_num_tokens)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/transformers/modeling_utils.py", line 1412, in _resize_token_embeddings
    new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/transformers/modeling_utils.py", line 1467, in _get_resized_embeddings
    new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/torch/nn/modules/module.py", line 989, in to
    return self._apply(convert)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/torch/nn/modules/module.py", line 664, in _apply
    param_applied = fn(param)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/torch/nn/modules/module.py", line 987, in convert
    return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 984.00 MiB (GPU 0; 47.50 GiB total capacity; 22.15 GiB already allocated; 68.88 MiB free; 22.16 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
Traceback (most recent call last):
  File "/home/opyate/Documents/code/github/dolly/training/trainer.py", line 423, in <module>
    main()
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/click/core.py", line 1130, in __call__
    return self.main(*args, **kwargs)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/click/core.py", line 1055, in main
    rv = self.invoke(ctx)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/click/core.py", line 1404, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/click/core.py", line 760, in invoke
    return __callback(*args, **kwargs)
  File "/home/opyate/Documents/code/github/dolly/training/trainer.py", line 395, in main
    train(**kwargs)
  File "/home/opyate/Documents/code/github/dolly/training/trainer.py", line 273, in train
    model, tokenizer = get_model_tokenizer(
  File "/home/opyate/Documents/code/github/dolly/training/trainer.py", line 209, in get_model_tokenizer
    model.resize_token_embeddings(len(tokenizer))
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/transformers/modeling_utils.py", line 1397, in resize_token_embeddings
    model_embeds = self._resize_token_embeddings(new_num_tokens)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/transformers/modeling_utils.py", line 1412, in _resize_token_embeddings
    new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/transformers/modeling_utils.py", line 1467, in _get_resized_embeddings
    new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/torch/nn/modules/module.py", line 989, in to
    return self._apply(convert)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/torch/nn/modules/module.py", line 664, in _apply
    param_applied = fn(param)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/torch/nn/modules/module.py", line 987, in convert
    return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 984.00 MiB (GPU 0; 47.50 GiB total capacity; 22.15 GiB already allocated; 68.88 MiB free; 22.16 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
2023-06-14 00:08:13 ERROR [__main__] main failed
Traceback (most recent call last):
  File "/home/opyate/Documents/code/github/dolly/training/trainer.py", line 423, in <module>
    main()
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/click/core.py", line 1130, in __call__
    return self.main(*args, **kwargs)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/click/core.py", line 1055, in main
    rv = self.invoke(ctx)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/click/core.py", line 1404, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/click/core.py", line 760, in invoke
    return __callback(*args, **kwargs)
  File "/home/opyate/Documents/code/github/dolly/training/trainer.py", line 395, in main
    train(**kwargs)
  File "/home/opyate/Documents/code/github/dolly/training/trainer.py", line 273, in train
    model, tokenizer = get_model_tokenizer(
  File "/home/opyate/Documents/code/github/dolly/training/trainer.py", line 209, in get_model_tokenizer
    model.resize_token_embeddings(len(tokenizer))
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/transformers/modeling_utils.py", line 1397, in resize_token_embeddings
    model_embeds = self._resize_token_embeddings(new_num_tokens)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/transformers/modeling_utils.py", line 1412, in _resize_token_embeddings
    new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/transformers/modeling_utils.py", line 1467, in _get_resized_embeddings
    new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/torch/nn/modules/module.py", line 989, in to
    return self._apply(convert)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/torch/nn/modules/module.py", line 664, in _apply
    param_applied = fn(param)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/torch/nn/modules/module.py", line 987, in convert
    return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 984.00 MiB (GPU 0; 47.50 GiB total capacity; 22.15 GiB already allocated; 68.88 MiB free; 22.16 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
Traceback (most recent call last):
  File "/home/opyate/Documents/code/github/dolly/training/trainer.py", line 423, in <module>
    main()
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/click/core.py", line 1130, in __call__
    return self.main(*args, **kwargs)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/click/core.py", line 1055, in main
    rv = self.invoke(ctx)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/click/core.py", line 1404, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/click/core.py", line 760, in invoke
    return __callback(*args, **kwargs)
  File "/home/opyate/Documents/code/github/dolly/training/trainer.py", line 395, in main
    train(**kwargs)
  File "/home/opyate/Documents/code/github/dolly/training/trainer.py", line 273, in train
    model, tokenizer = get_model_tokenizer(
  File "/home/opyate/Documents/code/github/dolly/training/trainer.py", line 209, in get_model_tokenizer
    model.resize_token_embeddings(len(tokenizer))
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/transformers/modeling_utils.py", line 1397, in resize_token_embeddings
    model_embeds = self._resize_token_embeddings(new_num_tokens)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/transformers/modeling_utils.py", line 1412, in _resize_token_embeddings
    new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/transformers/modeling_utils.py", line 1467, in _get_resized_embeddings
    new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/torch/nn/modules/module.py", line 989, in to
    return self._apply(convert)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/torch/nn/modules/module.py", line 664, in _apply
    param_applied = fn(param)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/torch/nn/modules/module.py", line 987, in convert
    return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 984.00 MiB (GPU 0; 47.50 GiB total capacity; 22.15 GiB already allocated; 68.88 MiB free; 22.16 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 80338) of binary: /home/opyate/anaconda3/envs/opyate-dolly-py39/bin/python3.9
Traceback (most recent call last):
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/bin/accelerate", line 8, in <module>
    sys.exit(main())
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/accelerate/commands/accelerate_cli.py", line 45, in main
    args.func(args)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/accelerate/commands/launch.py", line 960, in launch_command
    multi_gpu_launcher(args)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/accelerate/commands/launch.py", line 649, in multi_gpu_launcher
    distrib_run.run(args)
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/torch/distributed/run.py", line 753, in run
    elastic_launch(
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/torch/distributed/launcher/api.py", line 132, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/opyate/anaconda3/envs/opyate-dolly-py39/lib/python3.9/site-packages/torch/distributed/launcher/api.py", line 246, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
training/trainer.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2023-06-14_00:08:16
  host      : nil
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 80339)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2023-06-14_00:08:16
  host      : nil
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 80338)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
```