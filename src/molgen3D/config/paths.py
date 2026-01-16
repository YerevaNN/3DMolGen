/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 566, in launch_qwen3_pretrain
    job_config = config_manager.parse_args(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/config/manager.py", line 52, in parse_args
    launch_qwen3_pretrain(tyro.cli(QwenPretrainRunConfig))
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 566, in launch_qwen3_pretrain
    self._dict_to_dataclass(config_cls, toml_values)
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/config/manager.py", line 179, in _dict_to_dataclass
    job_config = config_manager.parse    raise TypeError(
TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'list'
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 615, in <module>
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 615, in <module>
    launch_qwen3_pretrain(tyro.cli(QwenPretrainRunConfig))
  File "/home/chem-project/aram-3dmolgen_args(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/config/manager.py", line 52, in parse_args
    result[f.name] = self._dict_to_dataclass(f.type, value)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/config/manager.py", line 182, in _dict_to_dataclass
    self._dict_to_dataclass(config_cls, toml_values)
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/config/manager.py", line 179, in _dict_to_dataclass
    return cls(**result)
           ^^^^^^^^^^^^^
  File "<string>", line 17, in __init__
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/config/custom_job_config.py", line 87, in __post_init__
    result[f.name] = self._dict_to_dataclass(f.type, value)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/config/manager.py", line 182, in _dict_to_dataclass
    _resolve_via(self.train_path_key, get_data_path)
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/config/custom_job_config.py", line 37, in _resolve_via
    return cls(**result)
           ^^^^^^^^^^^^^
  File "<string>", line 17, in __init__
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/config/custom_job_config.py", line 87, in __post_init__
    _resolve_via(self.train_path_key, get_data_path)
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/config/custom_job_config.py", line 37, in _resolve_via
    resolved = resolver(normalized)
               ^^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/config/paths.py", line 193, in get_data_path
    resolved = resolver(normalized)
               ^^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/config/paths.py", line 193, in get_data_path
    rel_candidates = _as_path_candidates(data_cfg[key])
          ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/pathlib.py", line 1164, in __init__
    super().__init__(*args)
  File "/usr/lib/python3.12/pathlib.py", line 373, in __init__
    raise TypeError(
TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'list'
    rel_candidates = _as_path_candidates(data_cfg[key])
          ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/pathlib.py", line 1164, in __init__
    super().__init__(*args)
  File "/usr/lib/python3.12/pathlib.py", line 373, in __init__
    raise TypeError(
TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'list'
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 615, in <module>
    launch_qwen3_pretrain(tyro.cli(QwenPretrainRunConfig))
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 566, in launch_qwen3_pretrain
    job_config = config_manager.parse_args(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/config/manager.py", line 52, in parse_args
    self._dict_to_dataclass(config_cls, toml_values)
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/config/manager.py", line 179, in _dict_to_dataclass
    result[f.name] = self._dict_to_dataclass(f.type, value)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/config/manager.py", line 182, in _dict_to_dataclass
    return cls(**result)
           ^^^^^^^^^^^^^
  File "<string>", line 17, in __init__
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/config/custom_job_config.py", line 87, in __post_init__
    _resolve_via(self.train_path_key, get_data_path)
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/config/custom_job_config.py", line 37, in _resolve_via
    resolved = resolver(normalized)
               ^^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/config/paths.py", line 193, in get_data_path
    rel_candidates = _as_path_candidates(data_cfg[key])
          ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/pathlib.py", line 1164, in __init__
    super().__init__(*args)
  File "/usr/lib/python3.12/pathlib.py", line 373, in __init__
    raise TypeError(
TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'list'
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 615, in <module>
    launch_qwen3_pretrain(tyro.cli(QwenPretrainRunConfig))
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 566, in launch_qwen3_pretrain
    job_config = config_manager.parse_args(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/config/manager.py", line 52, in parse_args
    self._dict_to_dataclass(config_cls, toml_values)
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/config/manager.py", line 179, in _dict_to_dataclass
    result[f.name] = self._dict_to_dataclass(f.type, value)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/config/manager.py", line 182, in _dict_to_dataclass
    return cls(**result)
           ^^^^^^^^^^^^^
  File "<string>", line 17, in __init__
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/config/custom_job_config.py", line 87, in __post_init__
    _resolve_via(self.train_path_key, get_data_path)
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/config/custom_job_config.py", line 37, in _resolve_via
    resolved = resolver(normalized)
               ^^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/config/paths.py", line 193, in get_data_path
    rel_candidates = _as_path_candidates(data_cfg[key])
          ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/pathlib.py", line 1164, in __init__
    super().__init__(*args)
  File "/usr/lib/python3.12/pathlib.py", line 373, in __init__
    raise TypeError(
TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'list'
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 615, in <module>
    launch_qwen3_pretrain(tyro.cli(QwenPretrainRunConfig))
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 566, in launch_qwen3_pretrain
    job_config = config_manager.parse_args(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/config/manager.py", line 52, in parse_args
    self._dict_to_dataclass(config_cls, toml_values)
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/config/manager.py", line 179, in _dict_to_dataclass
    result[f.name] = self._dict_to_dataclass(f.type, value)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/config/manager.py", line 182, in _dict_to_dataclass
    return cls(**result)
           ^^^^^^^^^^^^^
  File "<string>", line 17, in __init__
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/config/custom_job_config.py", line 87, in __post_init__
    _resolve_via(self.train_path_key, get_data_path)
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/config/custom_job_config.py", line 37, in _resolve_via
    resolved = resolver(normalized)
               ^^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/config/paths.py", line 193, in get_data_path
    rel_candidates = _as_path_candidates(data_cfg[key])
          ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/pathlib.py", line 1164, in __init__
    super().__init__(*args)
  File "/usr/lib/python3.12/pathlib.py", line 373, in __init__
    raise TypeError(
TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'list'
W0116 02:37:57.368000 844994 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 845116 closing signal SIGTERM
W0116 02:37:57.371000 844994 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 845117 closing signal SIGTERM
W0116 02:37:57.372000 844994 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 845119 closing signal SIGTERM
W0116 02:37:57.372000 844994 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 845120 closing signal SIGTERM
W0116 02:37:57.372000 844994 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 845122 closing signal SIGTERM
W0116 02:37:57.372000 844994 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 845123 closing signal SIGTERM
E0116 02:37:57.582000 844994 torch/distributed/elastic/multiprocessing/api.py:984] failed (exitcode: 1) local_rank: 2 (pid: 845118) of binary: /home/chem-project/aram-3dmolgen/3DMolGen/.venv/bin/python3
Traceback (most recent call last):
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/bin/torchrun", line 10, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/run.py", line 990, in main
    run(args)
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/run.py", line 981, in run
    elastic_launch(
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/launcher/api.py", line 170, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/launcher/api.py", line 317, in launch_agent

  rank      : 5 (local_rank: 5)
  exitcode  : 1 (pid: 845121)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2026-01-16_02:37:57
  host      : gpu01
  rank      : 0 (local_rank: 0)
  exitcode  : -15 (pid: 845116)
  error_file: <N/A>
  traceback : Signal 15 (SIGTERM) received by PID 845116
[3]:
  time      : 2026-01-16_02:37:57
  host      : gpu01
  rank      : 1 (local_rank: 1)
  exitcode  : -15 (pid: 845117)
  error_file: <N/A>
  traceback : Signal 15 (SIGTERM) received by PID 845117
[4]:
  time      : 2026-01-16_02:37:57
  host      : gpu01
  rank      : 3 (local_rank: 3)
  exitcode  : -15 (pid: 845119)
  error_file: <N/A>
  traceback : Signal 15 (SIGTERM) received by PID 845119
[5]:
  time      : 2026-01-16_02:37:57
  host      : gpu01
  rank      : 4 (local_rank: 4)
  exitcode  : -15 (pid: 845120)
  error_file: <N/A>
  traceback : Signal 15 (SIGTERM) received by PID 845120
[6]:
  time      : 2026-01-16_02:37:57
  host      : gpu01
  rank      : 6 (local_rank: 6)
  exitcode  : -15 (pid: 845122)
  error_file: <N/A>
  traceback : Signal 15 (SIGTERM) received by PID 845122
[7]:
  time      : 2026-01-16_02:37:57
  host      : gpu01
  rank      : 7 (local_rank: 7)
  exitcode  : -15 (pid: 845123)
  error_file: <N/A>
  traceback : Signal 15 (SIGTERM) received by PID 845123
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-01-16_02:37:57
  host      : gpu01
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 845118)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
(molgen3d) root@gpu01:/home/chem-project/mb-3dmolgen/3DMolGen# bash scripts/launch_torchtitan_qwen3.sh
Using description: qwen3_06b_pre_5e_48effb_4kseq
Run name: 260116-0241-af95-qwen3_06b_pre_5e_48effb_4kseq
W0116 02:41:13.328000 846088 torch/distributed/run.py:851]
W0116 02:41:13.328000 846088 torch/distributed/run.py:851] *****************************************
W0116 02:41:13.328000 846088 torch/distributed/run.py:851] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0116 02:41:13.328000 846088 torch/distributed/run.py:851] *****************************************
[titan] 2026-01-16 02:41:27,329 - root - INFO - Starting job: qwen3_06b_pre_5e_48effb_4kseq
[titan] 2026-01-16 02:41:27,350 - root - INFO - Starting job: qwen3_06b_pre_5e_48effb_4kseq
TOKENIZER_CHECK: sample1='[SMILES][CONFORMER][/SMILES][/CONFORMER]' -> ids=[151669, 151670, 151671, 151672] -> sample2=[151669, 151670, 151671, 151672] -> decoded='[SMILES][CONFORMER][/SMILES][/CONFORMER]'
[titan] 2026-01-16 02:41:27,399 - root - INFO - Starting job: qwen3_06b_pre_5e_48effb_4kseq
[titan] 2026-01-16 02:41:27,409 - root - INFO - Starting job: qwen3_06b_pre_5e_48effb_4kseq
[titan] 2026-01-16 02:41:27,434 - root - INFO - Starting job: qwen3_06b_pre_5e_48effb_4kseq
[titan] 2026-01-16 02:41:27,699 - root - INFO - Starting job: qwen3_06b_pre_5e_48effb_4kseq
[titan] 2026-01-16 02:41:27,795 - root - INFO - Starting job: qwen3_06b_pre_5e_48effb_4kseq
[titan] 2026-01-16 02:41:27,836 - root - INFO - Starting job: qwen3_06b_pre_5e_48effb_4kseq
[titan] 2026-01-16 02:41:28,949 - root - WARNING - ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be overridden to 3 based on job config
[titan] 2026-01-16 02:41:28,956 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=1, etp=1
[titan] 2026-01-16 02:41:28,963 - root - WARNING - ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be overridden to 3 based on job config
[titan] 2026-01-16 02:41:28,968 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=1, etp=1
[titan] 2026-01-16 02:41:28,983 - root - INFO - Successfully created meshes with active dimensions: ['batch', 'loss', 'fsdp', 'efsdp']
[titan] 2026-01-16 02:41:28,983 - root - INFO - Successfully created meshes with active dimensions: ['batch', 'loss', 'fsdp', 'efsdp']
[titan] 2026-01-16 02:41:28,983 - root - INFO - [GC] Initial GC collection took 0.00 seconds
[titan] 2026-01-16 02:41:28,984 - root - INFO - [GC] Initial GC collection took 0.00 seconds
[titan] 2026-01-16 02:41:29,127 - root - WARNING - ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be overridden to 3 based on job config
[titan] 2026-01-16 02:41:29,144 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=1, etp=1
[titan] 2026-01-16 02:41:29,171 - root - INFO - Successfully created meshes with active dimensions: ['batch', 'loss', 'fsdp', 'efsdp']
[titan] 2026-01-16 02:41:29,172 - root - INFO - [GC] Initial GC collection took 0.00 seconds
[titan] 2026-01-16 02:41:29,270 - root - WARNING - ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be overridden to 3 based on job config
[titan] 2026-01-16 02:41:29,277 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=1, etp=1
[titan] 2026-01-16 02:41:29,293 - root - WARNING - ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be overridden to 3 based on job config
[titan] 2026-01-16 02:41:29,302 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=1, etp=1
[titan] 2026-01-16 02:41:29,304 - root - INFO - Successfully created meshes with active dimensions: ['batch', 'loss', 'fsdp', 'efsdp']
[titan] 2026-01-16 02:41:29,304 - root - INFO - [GC] Initial GC collection took 0.00 seconds
[titan] 2026-01-16 02:41:29,327 - root - INFO - Successfully created meshes with active dimensions: ['batch', 'loss', 'fsdp', 'efsdp']
[titan] 2026-01-16 02:41:29,328 - root - INFO - [GC] Initial GC collection took 0.00 seconds
[titan] 2026-01-16 02:41:29,488 - root - WARNING - ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be overridden to 3 based on job config
[titan] 2026-01-16 02:41:29,493 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=1, etp=1
[titan] 2026-01-16 02:41:29,506 - root - WARNING - ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be overridden to 3 based on job config
[titan] 2026-01-16 02:41:29,510 - root - INFO - Successfully created meshes with active dimensions: ['batch', 'loss', 'fsdp', 'efsdp']
[titan] 2026-01-16 02:41:29,511 - root - INFO - [GC] Initial GC collection took 0.00 seconds
[titan] 2026-01-16 02:41:29,518 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=1, etp=1
[titan] 2026-01-16 02:41:29,522 - root - WARNING - ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be overridden to 3 based on job config
[titan] 2026-01-16 02:41:29,528 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=8, cp=1, tp=1, ep=1, etp=1
[titan] 2026-01-16 02:41:29,532 - root - INFO - Successfully created meshes with active dimensions: ['batch', 'loss', 'fsdp', 'efsdp']
[titan] 2026-01-16 02:41:29,533 - root - INFO - [GC] Initial GC collection took 0.00 seconds
[titan] 2026-01-16 02:41:29,540 - root - INFO - Successfully created meshes with active dimensions: ['batch', 'loss', 'fsdp', 'efsdp']
[titan] 2026-01-16 02:41:29,541 - root - INFO - [GC] Initial GC collection took 0.00 seconds
[titan] 2026-01-16 02:41:32,794 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-01-16 02:41:32,794 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-01-16 02:41:32,794 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-01-16 02:41:32,795 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-01-16 02:41:32,795 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-01-16 02:41:32,795 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-01-16 02:41:32,796 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-01-16 02:41:32,796 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-01-16 02:41:33,517 - root - INFO - Building molgen_qwen3 0.6Bwith {
  "_enforced": "This field is used to enforce all fields have defaults.",
  "dim": 1024,
  "n_layers": 28,
  "n_heads": 16,
  "n_kv_heads": 8,
  "vocab_size": 151936,
  "head_dim": 128,
  "hidden_dim": 3072,
  "norm_eps": 1e-06,
  "rope_theta": 1000000,
  "qk_norm": true,
  "max_seq_len": 4096,
  "depth_init": true,
  "attn_type": "sdpa",
  "attn_mask_type": "causal",
  "eos_id": 151645,
  "enable_weight_tying": true,
  "moe_enabled": false,
  "moe_inter_dim": 768,
  "moe_args": {
    "num_experts": 8,
    "num_shared_experts": 1,
    "score_func": "sigmoid",
    "route_norm": false,
    "route_scale": 1.0,
    "score_before_experts": true,
    "top_k": 1,
    "num_expert_groups": null,
    "num_limited_groups": null,
    "use_grouped_mm": true,
    "load_balance_coeff": 0.001,
    "_debug_force_load_balance": false
  }
}
[titan] 2026-01-16 02:41:33,555 - root - INFO - Building molgen_qwen3 0.6Bwith {
  "_enforced": "This field is used to enforce all fields have defaults.",
  "dim": 1024,
  "n_layers": 28,
  "n_heads": 16,
  "n_kv_heads": 8,
  "vocab_size": 151936,
  "head_dim": 128,
  "hidden_dim": 3072,
  "norm_eps": 1e-06,
  "rope_theta": 1000000,
  "qk_norm": true,
  "max_seq_len": 4096,
  "depth_init": true,
  "attn_type": "sdpa",
  "attn_mask_type": "causal",
  "eos_id": 151645,
  "enable_weight_tying": true,
  "moe_enabled": false,
  "moe_inter_dim": 768,
  "moe_args": {
    "num_experts": 8,
    "num_shared_experts": 1,
    "score_func": "sigmoid",
    "route_norm": false,
    "route_scale": 1.0,
    "score_before_experts": true,
    "top_k": 1,
    "num_expert_groups": null,
    "num_limited_groups": null,
    "use_grouped_mm": true,
    "load_balance_coeff": 0.001,
    "_debug_force_load_balance": false
  }
}
[titan] 2026-01-16 02:41:33,557 - root - INFO - Building molgen_qwen3 0.6Bwith {
  "_enforced": "This field is used to enforce all fields have defaults.",
  "dim": 1024,
  "n_layers": 28,
  "n_heads": 16,
  "n_kv_heads": 8,
  "vocab_size": 151936,
  "head_dim": 128,
  "hidden_dim": 3072,
  "norm_eps": 1e-06,
  "rope_theta": 1000000,
  "qk_norm": true,
  "max_seq_len": 4096,
  "depth_init": true,
  "attn_type": "sdpa",
  "attn_mask_type": "causal",
  "eos_id": 151645,
  "enable_weight_tying": true,
  "moe_enabled": false,
  "moe_inter_dim": 768,
  "moe_args": {
    "num_experts": 8,
    "num_shared_experts": 1,
    "score_func": "sigmoid",
    "route_norm": false,
    "route_scale": 1.0,
    "score_before_experts": true,
    "top_k": 1,
    "num_expert_groups": null,
    "num_limited_groups": null,
    "use_grouped_mm": true,
    "load_balance_coeff": 0.001,
    "_debug_force_load_balance": false
  }
}
[titan] 2026-01-16 02:41:33,560 - root - INFO - Building molgen_qwen3 0.6Bwith {
  "_enforced": "This field is used to enforce all fields have defaults.",
  "dim": 1024,
  "n_layers": 28,
  "n_heads": 16,
  "n_kv_heads": 8,
  "vocab_size": 151936,
  "head_dim": 128,
  "hidden_dim": 3072,
  "norm_eps": 1e-06,
  "rope_theta": 1000000,
  "qk_norm": true,
  "max_seq_len": 4096,
  "depth_init": true,
  "attn_type": "sdpa",
  "attn_mask_type": "causal",
  "eos_id": 151645,
  "enable_weight_tying": true,
  "moe_enabled": false,
  "moe_inter_dim": 768,
  "moe_args": {
    "num_experts": 8,
    "num_shared_experts": 1,
    "score_func": "sigmoid",
    "route_norm": false,
    "route_scale": 1.0,
    "score_before_experts": true,
    "top_k": 1,
    "num_expert_groups": null,
    "num_limited_groups": null,
    "use_grouped_mm": true,
    "load_balance_coeff": 0.001,
    "_debug_force_load_balance": false
  }
}
[titan] 2026-01-16 02:41:33,615 - root - INFO - Building molgen_qwen3 0.6Bwith {
  "_enforced": "This field is used to enforce all fields have defaults.",
  "dim": 1024,
  "n_layers": 28,
  "n_heads": 16,
  "n_kv_heads": 8,
  "vocab_size": 151936,
  "head_dim": 128,
  "hidden_dim": 3072,
  "norm_eps": 1e-06,
  "rope_theta": 1000000,
  "qk_norm": true,
  "max_seq_len": 4096,
  "depth_init": true,
  "attn_type": "sdpa",
  "attn_mask_type": "causal",
  "eos_id": 151645,
  "enable_weight_tying": true,
  "moe_enabled": false,
  "moe_inter_dim": 768,
  "moe_args": {
    "num_experts": 8,
    "num_shared_experts": 1,
    "score_func": "sigmoid",
    "route_norm": false,
    "route_scale": 1.0,
    "score_before_experts": true,
    "top_k": 1,
    "num_expert_groups": null,
    "num_limited_groups": null,
    "use_grouped_mm": true,
    "load_balance_coeff": 0.001,
    "_debug_force_load_balance": false
  }
}
[titan] 2026-01-16 02:41:33,634 - root - INFO - CUDA capacity: NVIDIA H100 80GB HBM3 with 79.18GiB memory
[titan] 2026-01-16 02:41:33,636 - root - INFO - CUDA capacity: NVIDIA H100 80GB HBM3 with 79.18GiB memory
[titan] 2026-01-16 02:41:33,636 - root - INFO - CUDA capacity: NVIDIA H100 80GB HBM3 with 79.18GiB memory
[titan] 2026-01-16 02:41:33,637 - root - INFO - CUDA capacity: NVIDIA H100 80GB HBM3 with 79.18GiB memory
[titan] 2026-01-16 02:41:33,637 - root - INFO - CUDA capacity: NVIDIA H100 80GB HBM3 with 79.18GiB memory
[titan] 2026-01-16 02:41:33,808 - root - INFO - Total parameter count: dense 751,632,384, sparse 0, active 751,632,384
[titan] 2026-01-16 02:41:33,808 - root - INFO - Model molgen_qwen3 0.6B size: 596,049,920 total parameters
[titan] 2026-01-16 02:41:33,810 - root - INFO - Total parameter count: dense 751,632,384, sparse 0, active 751,632,384
[titan] 2026-01-16 02:41:33,810 - root - INFO - Total parameter count: dense 751,632,384, sparse 0, active 751,632,384
[titan] 2026-01-16 02:41:33,810 - root - INFO - Total parameter count: dense 751,632,384, sparse 0, active 751,632,384
[titan] 2026-01-16 02:41:33,810 - root - INFO - Model molgen_qwen3 0.6B size: 596,049,920 total parameters
[titan] 2026-01-16 02:41:33,810 - root - INFO - Model molgen_qwen3 0.6B size: 596,049,920 total parameters
[titan] 2026-01-16 02:41:33,810 - root - INFO - Model molgen_qwen3 0.6B size: 596,049,920 total parameters
[titan] 2026-01-16 02:41:33,810 - root - INFO - Applied selective activation checkpointing to the model
[titan] 2026-01-16 02:41:33,811 - root - INFO - Applied selective activation checkpointing to the model
[titan] 2026-01-16 02:41:33,812 - root - INFO - Applied selective activation checkpointing to the model
[titan] 2026-01-16 02:41:33,812 - root - INFO - Applied selective activation checkpointing to the model
[titan] 2026-01-16 02:41:33,814 - root - INFO - Total parameter count: dense 751,632,384, sparse 0, active 751,632,384
[titan] 2026-01-16 02:41:33,814 - root - INFO - Model molgen_qwen3 0.6B size: 596,049,920 total parameters
[titan] 2026-01-16 02:41:33,815 - root - INFO - Applied selective activation checkpointing to the model
[titan] 2026-01-16 02:41:33,923 - root - INFO - Building molgen_qwen3 0.6Bwith {
  "_enforced": "This field is used to enforce all fields have defaults.",
  "dim": 1024,
  "n_layers": 28,
  "n_heads": 16,
  "n_kv_heads": 8,
  "vocab_size": 151936,
  "head_dim": 128,
  "hidden_dim": 3072,
  "norm_eps": 1e-06,
  "rope_theta": 1000000,
  "qk_norm": true,
  "max_seq_len": 4096,
  "depth_init": true,
  "attn_type": "sdpa",
  "attn_mask_type": "causal",
  "eos_id": 151645,
  "enable_weight_tying": true,
  "moe_enabled": false,
  "moe_inter_dim": 768,
  "moe_args": {
    "num_experts": 8,
    "num_shared_experts": 1,
    "score_func": "sigmoid",
    "route_norm": false,
    "route_scale": 1.0,
    "score_before_experts": true,
    "top_k": 1,
    "num_expert_groups": null,
    "num_limited_groups": null,
    "use_grouped_mm": true,
    "load_balance_coeff": 0.001,
    "_debug_force_load_balance": false
  }
}
[titan] 2026-01-16 02:41:33,930 - root - INFO - Applied FSDP to the model
[titan] 2026-01-16 02:41:33,932 - root - INFO - Applied FSDP to the model
[titan] 2026-01-16 02:41:33,932 - root - INFO - Applied FSDP to the model
[titan] 2026-01-16 02:41:33,934 - root - INFO - Applied FSDP to the model
[titan] 2026-01-16 02:41:33,937 - root - INFO - Applied FSDP to the model
[titan] 2026-01-16 02:41:33,947 - root - INFO - Building molgen_qwen3 0.6Bwith {
  "_enforced": "This field is used to enforce all fields have defaults.",
  "dim": 1024,
  "n_layers": 28,
  "n_heads": 16,
  "n_kv_heads": 8,
  "vocab_size": 151936,
  "head_dim": 128,
  "hidden_dim": 3072,
  "norm_eps": 1e-06,
  "rope_theta": 1000000,
  "qk_norm": true,
  "max_seq_len": 4096,
  "depth_init": true,
  "attn_type": "sdpa",
  "attn_mask_type": "causal",
  "eos_id": 151645,
  "enable_weight_tying": true,
  "moe_enabled": false,
  "moe_inter_dim": 768,
  "moe_args": {
    "num_experts": 8,
    "num_shared_experts": 1,
    "score_func": "sigmoid",
    "route_norm": false,
    "route_scale": 1.0,
    "score_before_experts": true,
    "top_k": 1,
    "num_expert_groups": null,
    "num_limited_groups": null,
    "use_grouped_mm": true,
    "load_balance_coeff": 0.001,
    "_debug_force_load_balance": false
  }
}
[titan] 2026-01-16 02:41:33,958 - root - INFO - Building molgen_qwen3 0.6Bwith {
  "_enforced": "This field is used to enforce all fields have defaults.",
  "dim": 1024,
  "n_layers": 28,
  "n_heads": 16,
  "n_kv_heads": 8,
  "vocab_size": 151936,
  "head_dim": 128,
  "hidden_dim": 3072,
  "norm_eps": 1e-06,
  "rope_theta": 1000000,
  "qk_norm": true,
  "max_seq_len": 4096,
  "depth_init": true,
  "attn_type": "sdpa",
  "attn_mask_type": "causal",
  "eos_id": 151645,
  "enable_weight_tying": true,
  "moe_enabled": false,
  "moe_inter_dim": 768,
  "moe_args": {
    "num_experts": 8,
    "num_shared_experts": 1,
    "score_func": "sigmoid",
    "route_norm": false,
    "route_scale": 1.0,
    "score_before_experts": true,
    "top_k": 1,
    "num_expert_groups": null,
    "num_limited_groups": null,
    "use_grouped_mm": true,
    "load_balance_coeff": 0.001,
    "_debug_force_load_balance": false
  }
}
[titan] 2026-01-16 02:41:33,964 - root - INFO - CUDA capacity: NVIDIA H100 80GB HBM3 with 79.18GiB memory
[titan] 2026-01-16 02:41:33,980 - root - INFO - CUDA capacity: NVIDIA H100 80GB HBM3 with 79.18GiB memory
[titan] 2026-01-16 02:41:34,074 - root - INFO - Total parameter count: dense 751,632,384, sparse 0, active 751,632,384
[titan] 2026-01-16 02:41:34,074 - root - INFO - Model molgen_qwen3 0.6B size: 596,049,920 total parameters
[titan] 2026-01-16 02:41:34,075 - root - INFO - Applied selective activation checkpointing to the model
[titan] 2026-01-16 02:41:34,079 - root - INFO - Total parameter count: dense 751,632,384, sparse 0, active 751,632,384
[titan] 2026-01-16 02:41:34,079 - root - INFO - Model molgen_qwen3 0.6B size: 596,049,920 total parameters
[titan] 2026-01-16 02:41:34,081 - root - INFO - Applied selective activation checkpointing to the model
[titan] 2026-01-16 02:41:34,182 - root - INFO - Applied FSDP to the model
[titan] 2026-01-16 02:41:34,198 - root - INFO - Applied FSDP to the model
[titan] 2026-01-16 02:41:34,929 - root - INFO - Peak FLOPS used for computing MFU: 9.890e+14
[titan] 2026-01-16 02:41:34,930 - root - INFO - CUDA memory usage for model: 0.20GiB(0.25%)
[titan] 2026-01-16 02:41:34,930 - root - INFO - Peak FLOPS used for computing MFU: 9.890e+14
[titan] 2026-01-16 02:41:34,931 - root - INFO - CUDA memory usage for model: 0.20GiB(0.25%)
[titan] 2026-01-16 02:41:34,932 - root - WARNING - model.safetensors.index.json not found at hf_assets_path: /home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/tokenizers/Qwen3_tokenizer_custom/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
[titan] 2026-01-16 02:41:34,934 - root - INFO - Peak FLOPS used for computing MFU: 9.890e+14
[titan] 2026-01-16 02:41:34,935 - root - WARNING - model.safetensors.index.json not found at hf_assets_path: /home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/tokenizers/Qwen3_tokenizer_custom/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
[titan] 2026-01-16 02:41:34,935 - root - INFO - CUDA memory usage for model: 0.20GiB(0.25%)
[titan] 2026-01-16 02:41:34,937 - root - INFO - Peak FLOPS used for computing MFU: 9.890e+14
[titan] 2026-01-16 02:41:34,937 - root - WARNING - model.safetensors.index.json not found at hf_assets_path: /home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/tokenizers/Qwen3_tokenizer_custom/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
[titan] 2026-01-16 02:41:34,938 - root - INFO - CUDA memory usage for model: 0.20GiB(0.25%)
[titan] 2026-01-16 02:41:34,941 - root - INFO - Peak FLOPS used for computing MFU: 9.890e+14
[titan] 2026-01-16 02:41:34,942 - root - INFO - CUDA memory usage for model: 0.20GiB(0.25%)
[titan] 2026-01-16 02:41:34,943 - root - WARNING - model.safetensors.index.json not found at hf_assets_path: /home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/tokenizers/Qwen3_tokenizer_custom/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
[titan] 2026-01-16 02:41:34,947 - root - WARNING - model.safetensors.index.json not found at hf_assets_path: /home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/tokenizers/Qwen3_tokenizer_custom/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
[titan] 2026-01-16 02:41:34,949 - root - INFO - Peak FLOPS used for computing MFU: 9.890e+14
[titan] 2026-01-16 02:41:34,950 - root - INFO - CUDA memory usage for model: 0.20GiB(0.25%)
[rank1]:[W116 02:41:34.870060172 ProcessGroupGloo.cpp:524] Warning: Unable to resolve hostname to a (local) address. Using the loopback address as fallback. Manually set the network interface to bind to with GLOO_SOCKET_IFNAME. (function operator())
[titan] 2026-01-16 02:41:34,954 - root - WARNING - model.safetensors.index.json not found at hf_assets_path: /home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/tokenizers/Qwen3_tokenizer_custom/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
[titan] 2026-01-16 02:41:34,959 - root - INFO - Peak FLOPS used for computing MFU: 9.890e+14
[titan] 2026-01-16 02:41:34,960 - root - INFO - CUDA memory usage for model: 0.20GiB(0.25%)
[titan] 2026-01-16 02:41:34,964 - root - WARNING - model.safetensors.index.json not found at hf_assets_path: /home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/tokenizers/Qwen3_tokenizer_custom/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
[rank3]:[W116 02:41:34.894905132 ProcessGroupGloo.cpp:524] Warning: Unable to resolve hostname to a (local) address. Using the loopback address as fallback. Manually set the network interface to bind to with GLOO_SOCKET_IFNAME. (function operator())
[rank4]:[W116 02:41:34.897414530 ProcessGroupGloo.cpp:524] Warning: Unable to resolve hostname to a (local) address. Using the loopback address as fallback. Manually set the network interface to bind to with GLOO_SOCKET_IFNAME. (function operator())
[rank2]:[W116 02:41:34.903261752 ProcessGroupGloo.cpp:524] Warning: Unable to resolve hostname to a (local) address. Using the loopback address as fallback. Manually set the network interface to bind to with GLOO_SOCKET_IFNAME. (function operator())
[rank7]:[W116 02:41:34.910551544 ProcessGroupGloo.cpp:524] Warning: Unable to resolve hostname to a (local) address. Using the loopback address as fallback. Manually set the network interface to bind to with GLOO_SOCKET_IFNAME. (function operator())
[rank5]:[W116 02:41:34.914243145 ProcessGroupGloo.cpp:524] Warning: Unable to resolve hostname to a (local) address. Using the loopback address as fallback. Manually set the network interface to bind to with GLOO_SOCKET_IFNAME. (function operator())
[rank6]:[W116 02:41:34.915818936 ProcessGroupGloo.cpp:524] Warning: Unable to resolve hostname to a (local) address. Using the loopback address as fallback. Manually set the network interface to bind to with GLOO_SOCKET_IFNAME. (function operator())
wandb: Currently logged in as: menuab_zxcv (menuab_team) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.23.1
wandb: Run data is saved locally in /home/chem-project/aram-3dmolgen/3DMolGen/wandb_runs/260116-0241-af95-qwen3_06b_pre_5e_48effb_4kseq/wandb/run-20260116_024138-af95
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run 260116-0241-af95-qwen3_06b_pre_5e_48effb_4kseq
wandb:  View project at https://wandb.ai/menuab_team/3dmolgen
wandb:  View run at https://wandb.ai/menuab_team/3dmolgen/runs/af95
[titan] 2026-01-16 02:41:42,200 - root - INFO - CUDA capacity: NVIDIA H100 80GB HBM3 with 79.18GiB memory
[titan] 2026-01-16 02:41:42,300 - root - INFO - Total parameter count: dense 751,632,384, sparse 0, active 751,632,384
[titan] 2026-01-16 02:41:42,300 - root - INFO - Model molgen_qwen3 0.6B size: 596,049,920 total parameters
[titan] 2026-01-16 02:41:42,301 - root - INFO - Tokenizer ready: total=151673 | base=151669 | added=4 | embedding rows=151936 | hidden=1024 | tied=True
[titan] 2026-01-16 02:41:42,304 - root - INFO - Total parameter count: dense 751,632,384, sparse 0, active 751,632,384
[titan] 2026-01-16 02:41:42,304 - root - INFO - MolGen Qwen3 summary: model=molgen_qwen3 flavor=0.6B params=596,049,920 vocab=151936 seq_len=4096 dtype=bfloat16
[titan] 2026-01-16 02:41:42,306 - root - INFO - Applied selective activation checkpointing to the model
[titan] 2026-01-16 02:41:42,436 - root - INFO - Applied FSDP to the model
[titan] 2026-01-16 02:41:42,436 - root - INFO - Initializing extra embeddings: init_mode=scratch base_vocab=151669 num_new_tokens=4
[titan] 2026-01-16 02:41:42,447 - root - INFO - Embedding weight details: type=DTensor shape=(151936, 1024) device=meta flat_param=False
[titan] 2026-01-16 02:41:42,447 - root - INFO - MolGen Qwen3 embeddings: rows=151936 base=151669 new=4 padded=151936
[titan] 2026-01-16 02:41:42,885 - root - INFO - Peak FLOPS used for computing MFU: 9.890e+14
[titan] 2026-01-16 02:41:42,886 - root - INFO - CUDA memory usage for model: 0.20GiB(0.25%)
[titan] 2026-01-16 02:41:42,890 - root - WARNING - model.safetensors.index.json not found at hf_assets_path: /home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/tokenizers/Qwen3_tokenizer_custom/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
[rank0]:[W116 02:41:42.851309701 ProcessGroupGloo.cpp:524] Warning: Unable to resolve hostname to a (local) address. Using the loopback address as fallback. Manually set the network interface to bind to with GLOO_SOCKET_IFNAME. (function operator())
[titan] 2026-01-16 02:41:42,950 - root - INFO - Checkpointing active. Checkpoints will be loaded from and saved to /nfs/h100/raid/chem/checkpoints/yerevann/qwen3_06b/260116-0241-af95-qwen3_06b_pre_5e_48effb_4kseq
[titan] 2026-01-16 02:41:42,950 - root - INFO - Mixed precision training is handled by fully_shard
[titan] 2026-01-16 02:41:42,953 - root - INFO - Checkpointing active. Checkpoints will be loaded from and saved to /nfs/h100/raid/chem/checkpoints/yerevann/qwen3_06b/260116-0241-af95-qwen3_06b_pre_5e_48effb_4kseq
[titan] 2026-01-16 02:41:42,953 - root - INFO - Mixed precision training is handled by fully_shard
[titan] 2026-01-16 02:41:42,954 - root - INFO - Checkpointing active. Checkpoints will be loaded from and saved to /nfs/h100/raid/chem/checkpoints/yerevann/qwen3_06b/260116-0241-af95-qwen3_06b_pre_5e_48effb_4kseq
[titan] 2026-01-16 02:41:42,954 - root - INFO - Mixed precision training is handled by fully_shard
[titan] 2026-01-16 02:41:42,954 - root - INFO - Checkpointing active. Checkpoints will be loaded from and saved to /nfs/h100/raid/chem/checkpoints/yerevann/qwen3_06b/260116-0241-af95-qwen3_06b_pre_5e_48effb_4kseq
[titan] 2026-01-16 02:41:42,955 - root - INFO - Mixed precision training is handled by fully_shard
[titan] 2026-01-16 02:41:42,955 - root - INFO - Checkpointing active. Checkpoints will be loaded from and saved to /nfs/h100/raid/chem/checkpoints/yerevann/qwen3_06b/260116-0241-af95-qwen3_06b_pre_5e_48effb_4kseq
[titan] 2026-01-16 02:41:42,955 - root - INFO - Mixed precision training is handled by fully_shard
[titan] 2026-01-16 02:41:42,956 - root - INFO - Checkpointing active. Checkpoints will be loaded from and saved to /nfs/h100/raid/chem/checkpoints/yerevann/qwen3_06b/260116-0241-af95-qwen3_06b_pre_5e_48effb_4kseq
[titan] 2026-01-16 02:41:42,956 - root - INFO - Mixed precision training is handled by fully_shard
[titan] 2026-01-16 02:41:42,956 - root - INFO - Checkpointing active. Checkpoints will be loaded from and saved to /nfs/h100/raid/chem/checkpoints/yerevann/qwen3_06b/260116-0241-af95-qwen3_06b_pre_5e_48effb_4kseq
[titan] 2026-01-16 02:41:42,956 - root - INFO - Mixed precision training is handled by fully_shard
[titan] 2026-01-16 02:41:42,956 - root - INFO - Checkpointing active. Checkpoints will be loaded from and saved to /nfs/h100/raid/chem/checkpoints/yerevann/qwen3_06b/260116-0241-af95-qwen3_06b_pre_5e_48effb_4kseq
[titan] 2026-01-16 02:41:42,957 - root - INFO - Mixed precision training is handled by fully_shard
[titan] 2026-01-16 02:41:43,288 - root - INFO - Trainer is initialized with local batch size 6, global batch size 48, gradient accumulation steps 1, sequence length 4096, total steps 150000 (warmup 200)
[titan] 2026-01-16 02:41:43,288 - root - INFO - Training starts at step 1
2026-01-16 02:41:43.330 | INFO     | molgen3D.training.pretraining.helpers.validator:__init__:106 - Loaded AutoTokenizer from /home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/tokenizers/Qwen3_tokenizer_custom for token resolution
2026-01-16 02:41:43.330 | INFO     | molgen3D.training.pretraining.helpers.validator:__init__:113 - Resolving conformer tokens with debug=True...
2026-01-16 02:41:43.330 | INFO     | molgen3D.training.pretraining.helpers.validator:__init__:121 - MolGenNumericalValidator tokenizer: type=HuggingFaceTokenizer, has_inner=True, inner_type=Tokenizer
2026-01-16 02:41:43.330 | INFO     | molgen3D.training.pretraining.helpers.validator:__init__:125 - MolGenNumericalValidator token resolution: conformer_start_id=151670, conformer_end_id=151672
[titan] 2026-01-16 02:41:43,331 - root - INFO - Trainer is initialized with local batch size 6, global batch size 48, gradient accumulation steps 1, sequence length 4096, total steps 150000 (warmup 200)
[titan] 2026-01-16 02:41:43,331 - root - INFO - Training starts at step 1
[titan] 2026-01-16 02:41:43,363 - root - INFO - Trainer is initialized with local batch size 6, global batch size 48, gradient accumulation steps 1, sequence length 4096, total steps 150000 (warmup 200)
[titan] 2026-01-16 02:41:43,364 - root - INFO - Training starts at step 1
[titan] 2026-01-16 02:41:43,366 - root - INFO - Trainer is initialized with local batch size 6, global batch size 48, gradient accumulation steps 1, sequence length 4096, total steps 150000 (warmup 200)
[titan] 2026-01-16 02:41:43,366 - root - INFO - Training starts at step 1
[titan] 2026-01-16 02:41:43,425 - root - INFO - Trainer is initialized with local batch size 6, global batch size 48, gradient accumulation steps 1, sequence length 4096, total steps 150000 (warmup 200)
[titan] 2026-01-16 02:41:43,426 - root - INFO - Training starts at step 1
[titan] 2026-01-16 02:41:43,457 - root - INFO - Trainer is initialized with local batch size 6, global batch size 48, gradient accumulation steps 1, sequence length 4096, total steps 150000 (warmup 200)
[titan] 2026-01-16 02:41:43,458 - root - INFO - Training starts at step 1
[titan] 2026-01-16 02:41:43,477 - root - INFO - Trainer is initialized with local batch size 6, global batch size 48, gradient accumulation steps 1, sequence length 4096, total steps 150000 (warmup 200)
[titan] 2026-01-16 02:41:43,478 - root - INFO - Training starts at step 1
[titan] 2026-01-16 02:41:43,495 - root - INFO - Trainer is initialized with local batch size 6, global batch size 48, gradient accumulation steps 1, sequence length 4096, total steps 150000 (warmup 200)
[titan] 2026-01-16 02:41:43,495 - root - INFO - Training starts at step 1
2026-01-16 02:41:48.023 | WARNING  | molgen3D.training.pretraining.dataprocessing.dataloader:maybe_log:215 - PREVIEW_SAMPLE rank=0 idx=1 len=65 ids=[151669, 34, 11295, 3025, 11730, 34, 59, 3706, 43504, 31, 9533, 34, 2376, 46, 43504, 19191, 39, 60, 16, 46, 43504, 31, 39, 9533, 8281, 43504, 19191, 39, 60, 17, 7612, 43504, 19191, 39, 9533, 46, 6620, 34, 31, 39, 9533, 46, 6620, 34, 31, 39, 60, 17, 46, 6620, 34, 19191, 39, 9533, 46, 6620, 34, 31, 39, 9533, 46, 6620, 34, 31, '...'] decoded=[SMILES]C/C(C)=C\CC[C@](C)(O[C@@H]1O[C@H](CO[C@@H]2OC[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@H]1O)[C@H]1CC[C@]2(C)[C@@H]1[C@H](O)C[C@@H]1[C@@]3(C)CC[C@H](O[C@@H]4O[C@H](CO)[C@@H](O)[C@H](O)[C@H]4O)C(C)(C)[C@@H]3CC[C@]12C[/SMILES][CONFORMER][C]<6.9532,2.9418,-2.4220>/[C]<5.5516,3.1620,-2.9207>([C]<5.4209,4.0536,-4.1189>)=[C]<4.4966,2.6113,-2.3327>\[C]<4.5539,1.7028,-1.1414>[C]<3.2105,1.6457,-0.4127>[C@]<3.1481,0.5399,0.6539>([C]<4.3529,0.6203,1.5962>)([O]<3.1427,-0.6974,-0.1189>[C@@H]<3.8268,-1.8170,0.3109>1[O]<5.1577,-1.7255,-0.1722>[C@H]<5.9845,-2.8140,0.1779>([C]<7.3974,-2.5631,-0.3579>[O]<8.0811,-1.5556,0.3626>[C@@H]<7.8525,-0.2433,-0.0917>2[O]<8.5811,0.0268,-1.2647>[C]<9.9865,-0.0256,-1.0749>[C@@H]<10.3985,1.0542,-0.0784>([O]<10.0048,2.3305,-0.5518>)[C@H]<9.7178,0.7615,1.2848>([O]<10.2102,-0.4136,1.8648>)[C@H]<8.1948,0.7118,1.0798>2[O]<7.6438,1.9845,0.8815>)[C@@H]<5.4255,-4.1204,-0.4009>([O]<6.2363,-5.1896,0.0045>)[C@H]<3.9875,-4.3005,0.0863>([O]<3.4628,-5.4663,-0.4994>)[C@H]<3.1441,-3.0738,-0.2712>1[O]<1.8662,-3.2902,0.2584>)[C@H]<1.8424,0.5775,1.5007>1[C]<1.8357,1.7574,2.5078>[C]<0.5109,2.5030,2.3081>[C@]<-0.4395,1.4239,1.7718>2([C]<-0.7504,0.5002,2.9660>)[C@@H]<0.4989,0.7354,0.7511>1[C@H]<-0.1846,-0.4858,0.1520>([O]<0.5888,-1.1570,-0.8339>)[C]<-1.4829,-0.0315,-0.5322>[C@@H]<-2.4320,0.6847,0.4316>1[C@@]<-3.8746,0.9006,-0.1330>3([C]<-3.8923,1.7269,-1.4261>)[C]<-4.4664,-0.4909,-0.4460>[C]<-5.9696,-0.4597,-0.6977>[C@H]<-6.7123,0.1314,0.4948>([O]<-8.1255,0.1660,0.2722>[C@@H]<-8.8188,-1.0362,0.4652>4[O]<-8.7837,-1.8863,-0.6429>[C@H]<-9.6395,-1.6107,-1.7509>([C]<-9.0556,-0.5611,-2.7167>[O]<-8.9822,0.7482,-2.2176>)[C@@H]<-11.0977,-1.3614,-1.3176>([O]<-11.6939,-2.5790,-0.9421>)[C@H]<-11.2254,-0.3387,-0.1709>([O]<-11.1254,0.9930,-0.5655>)[C@H]<-10.2391,-0.7089,0.9613>4[O]<-10.6742,-1.8598,1.6553>)[C]<-6.2397,1.5602,0.8466>([C]<-6.7230,2.5749,-0.1909>)([C]<-6.9081,1.9270,2.1831>)[C@@H]<-4.7017,1.4957,1.0389>3[C]<-4.0910,2.8124,1.5203>[C]<-2.6972,2.5585,2.0872>[C@]<-1.7414,1.9288,1.0607>12[C]<-1.4068,3.0157,0.0296>[/CONFORMER]<|endoftext|>[SMILES]Cc1ccc(F)c([C@H]2[C@@H]3C=CC[C@@H](c4cccc(-c5ccc(N(C)C)cc5)c4)[C@@H]3C(=O)N2Cc2ccccc2)c1[/SMILES][CONFORMER][C]<-6.9887,0.3091,-2.6899>[c]<-6.2658,0.5872,-1.4077>1[c]<-6.8746,1.3386,-0.4073>[c]<-6.2205,1.5926,0.7826>[c]<-4.9451,1.0918,0.9690>([F]<-4.3133,1.3524,2.1368>)[c]<-4.3033,0.3379,-0.0025>([C@H]<-2.9215,-0.2097,0.2158>2[C@@H]<-2.9213,-1.7443,0.4423>3[C]<-3.1020,-2.1205,1.8777>=[C]<-2.1663,-2.7201,2.5948>[C]<-0.8141,-3.0532,2.0497>[C@@H]<-0.4593,-2.1378,0.8725>([c]<0.8743,-2.4967,0.2709>4[c]<1.0916,-3.7383,-0.3138>[c]<2.3301,-4.0533,-0.8422>[c]<3.3654,-3.1396,-0.7940>[c]<3.1702,-1.8878,-0.2172>(-[c]<4.2575,-0.9033,-0.1649>5[c]<5.5605,-1.2760,0.1529>[c]<6.5809,-0.3508,0.2101>[c]<6.3495,1.0060,-0.0557>([N]<7.3600,1.9388,0.0433>([C]<8.7303,1.4904,0.1098>)[C]<7.1260,3.2899,-0.4047>)[c]<5.0395,1.3740,-0.3920>[c]<4.0254,0.4416,-0.4375>5)[c]<1.9183,-1.5865,0.3123>4)[C@@H]<-1.5919,-2.1901,-0.1697>3[C]<-1.3078,-1.1361,-1.2289>(=[O]<-0.5172,-1.2091,-2.1429>)[N]<-2.0899,-0.0647,-0.9608>[titan] 2026-01-16 02:42:36,857 - root - INFO - validate step:  1  loss: 12.4699  memory: 69.81GiB(88.17%)  tps: 74,600
[titan] 2026-01-16 02:42:36,857 - root - INFO - validate step:  1  loss: 12.4699  memory: 69.81GiB(88.17%)  tps: 74,599
[titan] 2026-01-16 02:42:36,857 - root - INFO - validate step:  1  loss: 12.4699  memory: 69.81GiB(88.17%)  tps: 74,599
[titan] 2026-01-16 02:42:36,857 - root - INFO - validate step:  1  loss: 12.4699  memory: 69.81GiB(88.17%)  tps: 74,599
[titan] 2026-01-16 02:42:36,857 - root - INFO - validate step:  1  loss: 12.4699  memory: 69.81GiB(88.17%)  tps: 74,601
[titan] 2026-01-16 02:42:36,858 - root - INFO - validate step:  1  loss: 12.4699  memory: 69.81GiB(88.17%)  tps: 74,598
[titan] 2026-01-16 02:42:36,859 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[titan] 2026-01-16 02:42:36,859 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[titan] 2026-01-16 02:42:36,860 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[titan] 2026-01-16 02:42:36,861 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[titan] 2026-01-16 02:42:36,861 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[titan] 2026-01-16 02:42:36,861 - root - INFO - validate step:  1  loss: 12.4699  memory: 69.81GiB(88.17%)  tps: 74,599
[titan] 2026-01-16 02:42:36,861 - root - INFO - validate step:  1  loss: 12.4699  memory: 69.81GiB(88.17%)  tps: 74,594
[titan] 2026-01-16 02:42:36,861 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[titan] 2026-01-16 02:42:36,863 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[titan] 2026-01-16 02:42:36,864 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/distributed/utils.py:396: UserWarning: Set timeout is now only supported for either nccl or gloo.
  torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)
/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/distributed/utils.py:396: UserWarning: Set timeout is now only supported for either nccl or gloo.
  torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)
/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/distributed/utils.py:396: UserWarning: Set timeout is now only supported for either nccl or gloo.
  torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)
/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/distributed/utils.py:396: UserWarning: Set timeout is now only supported for either nccl or gloo.
  torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)
/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/distributed/utils.py:396: UserWarning: Set timeout is now only supported for either nccl or gloo.
  torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)
/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/distributed/utils.py:396: UserWarning: Set timeout is now only supported for either nccl or gloo.
  torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)
/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/distributed/utils.py:396: UserWarning: Set timeout is now only supported for either nccl or gloo.
  torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)
/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/distributed/utils.py:396: UserWarning: Set timeout is now only supported for either nccl or gloo.
  torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)
[titan] 2026-01-16 02:42:48,277 - root - INFO - step: 20  loss: 11.8347  grad_norm:  4.6875  memory: 69.83GiB(88.19%)  tps: 40,890  tflops: 261.49  mfu: 26.44%
[titan] 2026-01-16 02:42:48,277 - root - INFO - step: 20  loss: 11.8347  grad_norm:  4.6875  memory: 69.83GiB(88.19%)  tps: 40,890  tflops: 261.49  mfu: 26.44%
[titan] 2026-01-16 02:42:48,277 - root - INFO - step: 20  loss: 11.8347  grad_norm:  4.6875  memory: 69.83GiB(88.19%)  tps: 40,892  tflops: 261.50  mfu: 26.44%
[titan] 2026-01-16 02:42:48,277 - root - INFO - step: 20  loss: 11.8347  grad_norm:  4.6875  memory: 69.83GiB(88.19%)  tps: 40,894  tflops: 261.51  mfu: 26.44%
[titan] 2026-01-16 02:42:48,277 - root - INFO - step: 20  loss: 11.8347  grad_norm:  4.6875  memory: 69.83GiB(88.19%)  tps: 40,891  tflops: 261.49  mfu: 26.44%
[titan] 2026-01-16 02:42:48,278 - root - INFO - step: 20  loss: 11.8347  grad_norm:  4.6875  memory: 69.83GiB(88.19%)  tps: 40,904  tflops: 261.57  mfu: 26.45%
[titan] 2026-01-16 02:42:48,278 - root - INFO - step: 20  loss: 11.8347  grad_norm:  4.6875  memory: 69.83GiB(88.19%)  tps: 40,891  tflops: 261.49  mfu: 26.44%
[titan] 2026-01-16 02:42:48,278 - root - INFO - step: 20  loss: 11.8347  grad_norm:  4.6875  memory: 69.83GiB(88.19%)  tps: 40,906  tflops: 261.59  mfu: 26.45%
[titan] 2026-01-16 02:42:59,803 - root - INFO - step: 40  loss:  9.3507  grad_norm: 11.4375  memory: 69.83GiB(88.19%)  tps: 42,647  tflops: 272.72  mfu: 27.58%
[titan] 2026-01-16 02:42:59,803 - root - INFO - step: 40  loss:  9.3507  grad_norm: 11.4375  memory: 69.83GiB(88.19%)  tps: 42,646  tflops: 272.72  mfu: 27.58%
[titan] 2026-01-16 02:42:59,803 - root - INFO - step: 40  loss:  9.3507  grad_norm: 11.4375  memory: 69.83GiB(88.19%)  tps: 42,646  tflops: 272.72  mfu: 27.58%
[titan] 2026-01-16 02:42:59,803 - root - INFO - step: 40  loss:  9.3507  grad_norm: 11.4375  memory: 69.83GiB(88.19%)  tps: 42,647  tflops: 272.72  mfu: 27.58%
[titan] 2026-01-16 02:42:59,803 - root - INFO - step: 40  loss:  9.3507  grad_norm: 11.4375  memory: 69.83GiB(88.19%)  tps: 42,647  tflops: 272.72  mfu: 27.58%
[titan] 2026-01-16 02:42:59,804 - root - INFO - step: 40  loss:  9.3507  grad_norm: 11.4375  memory: 69.83GiB(88.19%)  tps: 42,648  tflops: 272.73  mfu: 27.58%
[titan] 2026-01-16 02:42:59,804 - root - INFO - step: 40  loss:  9.3507  grad_norm: 11.4375  memory: 69.83GiB(88.19%)  tps: 42,648  tflops: 272.73  mfu: 27.58%
[titan] 2026-01-16 02:42:59,804 - root - INFO - step: 40  loss:  9.3507  grad_norm: 11.4375  memory: 69.83GiB(88.19%)  tps: 42,649  tflops: 272.74  mfu: 27.58%
[titan] 2026-01-16 02:43:05,602 - root - INFO - [GC] Performing periodic GC collection took 0.69 seconds
[titan] 2026-01-16 02:43:05,637 - root - INFO - [GC] Performing periodic GC collection took 0.72 seconds
[titan] 2026-01-16 02:43:05,647 - root - INFO - [GC] Performing periodic GC collection took 0.73 seconds
[titan] 2026-01-16 02:43:05,677 - root - INFO - [GC] Performing periodic GC collection took 0.76 seconds
[titan] 2026-01-16 02:43:05,679 - root - INFO - [GC] Performing periodic GC collection took 0.76 seconds
[titan] 2026-01-16 02:43:05,819 - root - INFO - [GC] Performing periodic GC collection took 0.90 seconds
[titan] 2026-01-16 02:43:06,209 - root - INFO - [GC] Performing periodic GC collection took 1.28 seconds
[titan] 2026-01-16 02:43:06,419 - root - INFO - [GC] Performing periodic GC collection took 1.50 seconds
[titan] 2026-01-16 02:43:12,734 - root - INFO - step: 60  loss:  4.9399  grad_norm:  6.1875  memory: 69.83GiB(88.19%)  tps: 38,012  tflops: 243.08  mfu: 24.58%
[titan] 2026-01-16 02:43:12,734 - root - INFO - step: 60  loss:  4.9399  grad_norm:  6.1875  memory: 69.83GiB(88.19%)  tps: 38,012  tflops: 243.08  mfu: 24.58%
[titan] 2026-01-16 02:43:12,734 - root - INFO - step: 60  loss:  4.9399  grad_norm:  6.1875  memory: 69.83GiB(88.19%)  tps: 38,012  tflops: 243.08  mfu: 24.58%
[titan] 2026-01-16 02:43:12,734 - root - INFO - step: 60  loss:  4.9399  grad_norm:  6.1875  memory: 69.83GiB(88.19%)  tps: 38,012  tflops: 243.08  mfu: 24.58%
[titan] 2026-01-16 02:43:12,734 - root - INFO - step: 60  loss:  4.9399  grad_norm:  6.1875  memory: 69.83GiB(88.19%)  tps: 38,014  tflops: 243.09  mfu: 24.58%
[titan] 2026-01-16 02:43:12,735 - root - INFO - step: 60  loss:  4.9399  grad_norm:  6.1875  memory: 69.83GiB(88.19%)  tps: 38,013  tflops: 243.09  mfu: 24.58%
[titan] 2026-01-16 02:43:12,735 - root - INFO - step: 60  loss:  4.9399  grad_norm:  6.1875  memory: 69.83GiB(88.19%)  tps: 38,013  tflops: 243.09  mfu: 24.58%
[titan] 2026-01-16 02:43:12,735 - root - INFO - step: 60  loss:  4.9399  grad_norm:  6.1875  memory: 69.83GiB(88.19%)  tps: 38,015  tflops: 243.10  mfu: 24.58%
[titan] 2026-01-16 02:43:24,307 - root - INFO - step: 80  loss:  3.0349  grad_norm:  1.9453  memory: 69.83GiB(88.19%)  tps: 42,475  tflops: 271.62  mfu: 27.46%
[titan] 2026-01-16 02:43:24,307 - root - INFO - step: 80  loss:  3.0349  grad_norm:  1.9453  memory: 69.83GiB(88.19%)  tps: 42,475  tflops: 271.62  mfu: 27.46%
[titan] 2026-01-16 02:43:24,307 - root - INFO - step: 80  loss:  3.0349  grad_norm:  1.9453  memory: 69.83GiB(88.19%)  tps: 42,475  tflops: 271.62  mfu: 27.46%
[titan] 2026-01-16 02:43:24,307 - root - INFO - step: 80  loss:  3.0349  grad_norm:  1.9453  memory: 69.83GiB(88.19%)  tps: 42,475  tflops: 271.62  mfu: 27.46%
[titan] 2026-01-16 02:43:24,307 - root - INFO - step: 80  loss:  3.0349  grad_norm:  1.9453  memory: 69.83GiB(88.19%)  tps: 42,476  tflops: 271.63  mfu: 27.47%
[titan] 2026-01-16 02:43:24,307 - root - INFO - step: 80  loss:  3.0349  grad_norm:  1.9453  memory: 69.83GiB(88.19%)  tps: 42,475  tflops: 271.62  mfu: 27.46%
[titan] 2026-01-16 02:43:24,308 - root - INFO - step: 80  loss:  3.0349  grad_norm:  1.9453  memory: 69.83GiB(88.19%)  tps: 42,474  tflops: 271.62  mfu: 27.46%
[titan] 2026-01-16 02:43:24,308 - root - INFO - step: 80  loss:  3.0349  grad_norm:  1.9453  memory: 69.83GiB(88.19%)  tps: 42,478  tflops: 271.64  mfu: 27.47%
[titan] 2026-01-16 02:43:35,249 - root - INFO - [GC] Performing periodic GC collection took 0.05 seconds
[titan] 2026-01-16 02:43:35,252 - root - INFO - [GC] Performing periodic GC collection took 0.05 seconds
[titan] 2026-01-16 02:43:35,256 - root - INFO - [GC] Performing periodic GC collection took 0.06 seconds
[titan] 2026-01-16 02:43:35,267 - root - INFO - [GC] Performing periodic GC collection took 0.06 seconds
[titan] 2026-01-16 02:43:35,273 - root - INFO - [GC] Performing periodic GC collection took 0.07 seconds
[titan] 2026-01-16 02:43:35,277 - root - INFO - [GC] Performing periodic GC collection took 0.07 seconds
[titan] 2026-01-16 02:43:35,284 - root - INFO - [GC] Performing periodic GC collection took 0.08 seconds
[titan] 2026-01-16 02:43:35,285 - root - INFO - [GC] Performing periodic GC collection took 0.08 seconds
[titan] 2026-01-16 02:43:35,918 - root - INFO - step: 100  loss:  2.3367  grad_norm:  0.4648  memory: 69.83GiB(88.19%)  tps: 42,332  tflops: 270.71  mfu: 27.37%
[titan] 2026-01-16 02:43:35,918 - root - INFO - step: 100  loss:  2.3367  grad_norm:  0.4648  memory: 69.83GiB(88.19%)  tps: 42,332  tflops: 270.71  mfu: 27.37%
[titan] 2026-01-16 02:43:35,919 - root - INFO - step: 100  loss:  2.3367  grad_norm:  0.4648  memory: 69.83GiB(88.19%)  tps: 42,333  tflops: 270.71  mfu: 27.37%
[titan] 2026-01-16 02:43:35,919 - root - INFO - step: 100  loss:  2.3367  grad_norm:  0.4648  memory: 69.83GiB(88.19%)  tps: 42,331  tflops: 270.70  mfu: 27.37%
[titan] 2026-01-16 02:43:35,919 - root - INFO - step: 100  loss:  2.3367  grad_norm:  0.4648  memory: 69.83GiB(88.19%)  tps: 42,332  tflops: 270.71  mfu: 27.37%
[titan] 2026-01-16 02:43:35,919 - root - INFO - step: 100  loss:  2.3367  grad_norm:  0.4648  memory: 69.83GiB(88.19%)  tps: 42,333  tflops: 270.71  mfu: 27.37%
[titan] 2026-01-16 02:43:35,919 - root - INFO - step: 100  loss:  2.3367  grad_norm:  0.4648  memory: 69.83GiB(88.19%)  tps: 42,334  tflops: 270.72  mfu: 27.37%
[titan] 2026-01-16 02:43:35,920 - root - INFO - step: 100  loss:  2.3367  grad_norm:  0.4648  memory: 69.83GiB(88.19%)  tps: 42,338  tflops: 270.75  mfu: 27.38%
^CW0116 02:43:46.156000 846088 torch/distributed/elastic/agent/server/api.py:739] Received 2 death signal, shutting down workers
W0116 02:43:46.167000 846088 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 846210 closing signal SIGINT
W0116 02:43:46.170000 846088 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 846211 closing signal SIGINT
W0116 02:43:46.170000 846088 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 846212 closing signal SIGINT
W0116 02:43:46.170000 846088 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 846213 closing signal SIGINT
W0116 02:43:46.170000 846088 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 846214 closing signal SIGINT
W0116 02:43:46.170000 846088 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 846215 closing signal SIGINT
W0116 02:43:46.171000 846088 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 846216 closing signal SIGINT
W0116 02:43:46.171000 846088 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 846217 closing signal SIGINT
Traceback (most recent call last):
[rank5]: Traceback (most recent call last):
[rank5]:   File "<frozen runpy>", line 198, in _run_module_as_main
[rank5]:   File "<frozen runpy>", line 88, in _run_code
[rank5]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 615, in <module>
[rank5]:     launch_qwen3_pretrain(tyro.cli(QwenPretrainRunConfig))
[rank5]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 602, in launch_qwen3_pretrain
[rank5]:     trainer.train()
[rank5]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
[rank5]:     return f(*args, **kwargs)
[rank5]:            ^^^^^^^^^^^^^^^^^^
[rank5]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 669, in train
[rank5]:     self.train_step(data_iterator)
[rank5]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 567, in train_step
[rank5]:     input_dict, labels = next(data_iterator)
[rank5]:                          ^^^^^^^^^^^^^^^^^^^
[rank5]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 416, in batch_generator
[rank5]:     input_dict[k] = v.to(device_type)
[rank5]:                     ^^^^^^^^^^^^^^^^^
[rank5]: KeyboardInterrupt
[rank6]: Traceback (most recent call last):
[rank6]:   File "<frozen runpy>", line 198, in _run_module_as_main
[rank6]:   File "<frozen runpy>", line 88, in _run_code
[rank6]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 615, in <module>
[rank6]:     launch_qwen3_pretrain(tyro.cli(QwenPretrainRunConfig))
[rank6]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 602, in launch_qwen3_pretrain
[rank6]:     trainer.train()
[rank6]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
[rank6]:     return f(*args, **kwargs)
[rank6]:            ^^^^^^^^^^^^^^^^^^
[rank6]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 669, in train
[rank6]:     self.train_step(data_iterator)
[rank6]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 567, in train_step
[rank6]:     input_dict, labels = next(data_iterator)
[rank6]:                          ^^^^^^^^^^^^^^^^^^^
[rank6]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 416, in batch_generator
[rank6]:     input_dict[k] = v.to(device_type)
[rank6]:                     ^^^^^^^^^^^^^^^^^
[rank6]: KeyboardInterrupt
[rank2]: Traceback (most recent call last):
[rank2]:   File "<frozen runpy>", line 198, in _run_module_as_main
[rank2]:   File "<frozen runpy>", line 88, in _run_code
[rank2]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 615, in <module>
[rank2]:     launch_qwen3_pretrain(tyro.cli(QwenPretrainRunConfig))
[rank2]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 602, in launch_qwen3_pretrain
[rank2]:     trainer.train()
[rank2]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
[rank2]:     return f(*args, **kwargs)
[rank2]:            ^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 669, in train
[rank2]:     self.train_step(data_iterator)
[rank2]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 567, in train_step
[rank2]:     input_dict, labels = next(data_iterator)
[rank2]:                          ^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 416, in batch_generator
[rank2]:     input_dict[k] = v.to(device_type)
[rank2]:                     ^^^^^^^^^^^^^^^^^
[rank2]: KeyboardInterrupt
[rank1]: Traceback (most recent call last):
[rank1]:   File "<frozen runpy>", line 198, in _run_module_as_main
[rank1]:   File "<frozen runpy>", line 88, in _run_code
[rank1]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 615, in <module>
[rank1]:     launch_qwen3_pretrain(tyro.cli(QwenPretrainRunConfig))
[rank1]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 602, in launch_qwen3_pretrain
[rank1]:     trainer.train()
[rank1]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
[rank1]:     return f(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 669, in train
[rank1]:     self.train_step(data_iterator)
[rank1]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 567, in train_step
[rank1]:     input_dict, labels = next(data_iterator)
[rank1]:                          ^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 416, in batch_generator
[rank1]:     input_dict[k] = v.to(device_type)
[rank1]:                     ^^^^^^^^^^^^^^^^^
[rank1]: KeyboardInterrupt
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 615, in <module>
    launch_qwen3_pretrain(tyro.cli(QwenPretrainRunConfig))
[rank7]: Traceback (most recent call last):
[rank7]:   File "<frozen runpy>", line 198, in _run_module_as_main
[rank7]:   File "<frozen runpy>", line 88, in _run_code
[rank7]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 615, in <module>
[rank7]:     launch_qwen3_pretrain(tyro.cli(QwenPretrainRunConfig))
[rank7]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 602, in launch_qwen3_pretrain
[rank7]:     trainer.train()
[rank7]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
[rank7]:     return f(*args, **kwargs)
[rank7]:            ^^^^^^^^^^^^^^^^^^
[rank7]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 669, in train
[rank7]:     self.train_step(data_iterator)
[rank7]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 567, in train_step
[rank7]:     input_dict, labels = next(data_iterator)
[rank7]:                          ^^^^^^^^^^^^^^^^^^^
[rank7]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 416, in batch_generator
[rank7]:     input_dict[k] = v.to(device_type)
[rank7]:                     ^^^^^^^^^^^^^^^^^
[rank7]: KeyboardInterrupt
[rank4]: Traceback (most recent call last):
[rank4]:   File "<frozen runpy>", line 198, in _run_module_as_main
[rank4]:   File "<frozen runpy>", line 88, in _run_code
[rank4]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 615, in <module>
[rank4]:     launch_qwen3_pretrain(tyro.cli(QwenPretrainRunConfig))
[rank4]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 602, in launch_qwen3_pretrain
[rank4]:     trainer.train()
[rank4]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
[rank4]:     return f(*args, **kwargs)
[rank4]:            ^^^^^^^^^^^^^^^^^^
[rank4]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 669, in train
[rank4]:     self.train_step(data_iterator)
[rank4]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 567, in train_step
[rank4]:     input_dict, labels = next(data_iterator)
[rank4]:                          ^^^^^^^^^^^^^^^^^^^
[rank4]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 416, in batch_generator
[rank4]:     input_dict[k] = v.to(device_type)
[rank4]:                     ^^^^^^^^^^^^^^^^^
[rank4]: KeyboardInterrupt
[rank3]: Traceback (most recent call last):
[rank3]:   File "<frozen runpy>", line 198, in _run_module_as_main
[rank3]:   File "<frozen runpy>", line 88, in _run_code
[rank3]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 615, in <module>
[rank3]:     launch_qwen3_pretrain(tyro.cli(QwenPretrainRunConfig))
[rank3]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 602, in launch_qwen3_pretrain
[rank3]:     trainer.train()
[rank3]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
[rank3]:     return f(*args, **kwargs)
[rank3]:            ^^^^^^^^^^^^^^^^^^
[rank3]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 669, in train
[rank3]:     self.train_step(data_iterator)
[rank3]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 567, in train_step
[rank3]:     input_dict, labels = next(data_iterator)
[rank3]:                          ^^^^^^^^^^^^^^^^^^^
[rank3]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 416, in batch_generator
[rank3]:     input_dict[k] = v.to(device_type)
[rank3]:                     ^^^^^^^^^^^^^^^^^
[rank3]: KeyboardInterrupt
  File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 602, in launch_qwen3_pretrain
    trainer.train()
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 669, in train
    self.train_step(data_iterator)
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 567, in train_step
    input_dict, labels = next(data_iterator)
                         ^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 416, in batch_generator
    input_dict[k] = v.to(device_type)
                    ^^^^^^^^^^^^^^^^^
KeyboardInterrupt
[rank0]: Traceback (most recent call last):
[rank0]:   File "<frozen runpy>", line 198, in _run_module_as_main
[rank0]:   File "<frozen runpy>", line 88, in _run_code
[rank0]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 615, in <module>
[rank0]:     launch_qwen3_pretrain(tyro.cli(QwenPretrainRunConfig))
[rank0]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/src/molgen3D/training/pretraining/torchtitan_runner.py", line 602, in launch_qwen3_pretrain
[rank0]:     trainer.train()
[rank0]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
[rank0]:     return f(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 669, in train
[rank0]:     self.train_step(data_iterator)
[rank0]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 567, in train_step
[rank0]:     input_dict, labels = next(data_iterator)
[rank0]:                          ^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torchtitan/train.py", line 416, in batch_generator
[rank0]:     input_dict[k] = v.to(device_type)
[rank0]:                     ^^^^^^^^^^^^^^^^^
[rank0]: KeyboardInterrupt
^CW0116 02:43:46.373000 846088 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 846210 closing signal SIGTERM
W0116 02:43:46.373000 846088 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 846211 closing signal SIGTERM
W0116 02:43:46.374000 846088 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 846212 closing signal SIGTERM
W0116 02:43:46.374000 846088 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 846213 closing signal SIGTERM
W0116 02:43:46.375000 846088 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 846214 closing signal SIGTERM
W0116 02:43:46.376000 846088 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 846215 closing signal SIGTERM
W0116 02:43:46.376000 846088 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 846216 closing signal SIGTERM
W0116 02:43:46.377000 846088 torch/distributed/elastic/multiprocessing/api.py:1010] Sending process 846217 closing signal SIGTERM
Traceback (most recent call last):
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/agent/server/api.py", line 731, in run
    result = self._invoke_run(role)
             ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/agent/server/api.py", line 908, in _invoke_run
    time.sleep(monitor_interval)
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/api.py", line 86, in _terminate_process_handler
    raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)
torch.distributed.elastic.multiprocessing.api.SignalException: Process 846088 got signal: 2

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/bin/torchrun", line 10, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 362, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/run.py", line 990, in main
    run(args)
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/run.py", line 981, in run
    elastic_launch(
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/launcher/api.py", line 170, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/launcher/api.py", line 308, in launch_agent
    result = agent.run()
             ^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/metrics/api.py", line 134, in wrapper
    result = f(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/agent/server/api.py", line 740, in run
    self._shutdown(e.sigval)
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/agent/server/local_elastic_agent.py", line 416, in _shutdown
    self._pcontext.close(death_sig)
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/api.py", line 659, in close
    self._close(death_sig=death_sig, timeout=timeout)
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/api.py", line 1022, in _close
    handler.proc.wait(time_to_wait)
  File "/usr/lib/python3.12/subprocess.py", line 1264, in wait
    return self._wait(timeout=timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/subprocess.py", line 2047, in _wait
    time.sleep(delay)
  File "/home/chem-project/aram-3dmolgen/3DMolGen/.venv/lib/python3.12/site-packages/torch/distributed/elastic/multiprocessing/api.py", line 86, in _terminate_process_handler
    raise SignalException(f"Process {os.getpid()} got signal: {sigval}", sigval=sigval)
torch.distributed.elastic.multiprocessing.api.SignalException: Process 846088 got signal: 2
^C
(molgen3d) root@gpu01:/home/chem-project/mb-3dmolgen/3DMolGen# nvito[
> ^C
(molgen3d) root@gpu01:/home/chem-project/mb-3dmolgen/3DMolGen# nvitop
Fri Jan 16 02:47:44 2026
╒═════════════════════════════════════════════════════════════════════════════╕
│ NVITOP 1.6.1      Driver Version: 580.105.08      CUDA Driver Version: 13.0 │
├───────────────────────────────┬──────────────────────┬──────────────────────┤
│ GPU  Name        Persistence-M│ Bus-Id        Disp.A │ MIG M.   Uncorr. ECC │
│ Fan  Temp  Perf  Pwr:Usage/Cap│         Memory-Usage │ GPU-Util  Compute M. │
╞═══════════════════════════════╪══════════════════════╪══════════════════════╪═════════════════════════════╤══════════════════════════════╕
│   0  H100 80GB HBM3        On │ 00000000:18:00.0 Off │ Disabled           0 │ MEM: ▏ 0B              0.0% │ MBW: ▏ 0%          @ 2619MHz │
  GNU nano 7.2                                                                                      src/molgen3D/config/paths.py
from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
import importlib.resources as pkg_resources
import copy
import yaml


_ENV_REPO_ROOT = os.environ.get("MOLGEN3D_REPO_ROOT")
_ENV_PROJECT_ROOT = os.environ.get("MOLGEN3D_PROJECT_ROOT")
_CANDIDATE_ROOT = (
    Path(_ENV_REPO_ROOT).expanduser().resolve()
    if _ENV_REPO_ROOT
    else Path(__file__).resolve().parents[3]
)
if not (_CANDIDATE_ROOT / "src" / "molgen3D").exists():
    cwd = Path.cwd().resolve()
    if (cwd / "src" / "molgen3D").exists():
        _CANDIDATE_ROOT = cwd
REPO_ROOT = _CANDIDATE_ROOT

# Keys that should use geom_data_root instead of data_root
GEOM_DATA_KEYS = {
    "rdkit_folder",
    "test_mols",
    "drugs_summary",
    "conformers_train",
    "conformers_valid",
    "conformers_test",
    "pretokenized_prompts",
    "validation_pickle",
    "binned_conformers_train",
    "binned_conformers_valid",
    "binned_conformers_test",
    "filtered_conformers_train",
    "filtered_conformers_valid",
    "filtered_conformers_test",
}


def _path_candidate_roots() -> list[Path]:
    """Return ordered roots used when resolving relative paths."""
    roots: list[Path] = []

    def _add_root(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in roots:
            roots.append(resolved)

    if _ENV_PROJECT_ROOT:
        _add_root(Path(_ENV_PROJECT_ROOT).expanduser())

    _add_root(REPO_ROOT)
    for ancestor in REPO_ROOT.parents[:2]:
        _add_root(ancestor)
                                                                                                      [ Read 527 lines ]
^G Help          ^O Write Out     ^W Where Is      ^K Cut           ^T Execute       ^C Location      M-U Undo         M-A Set Mark     M-] To Bracket   M-Q Previous     ^B Back          ^◂ Prev Word     ^A Home
^X Exit          ^R Read File     ^\ Replace       ^U Paste         ^J Justify       ^/ Go To Line    M-E Redo         M-6 Copy         ^Q Where Was     M-W Next         ^F Forward       ^▸ Next Word     ^E End
[2] 0:bash  1:ssh* 2:ssh- 3:ssh  4:bash  5:bash  6:bash                                                                                                                                        "bcm11-headnode" 02:49 16-Jan-26from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
import importlib.resources as pkg_resources
import copy
import yaml


_ENV_REPO_ROOT = os.environ.get("MOLGEN3D_REPO_ROOT")
_ENV_PROJECT_ROOT = os.environ.get("MOLGEN3D_PROJECT_ROOT")
_CANDIDATE_ROOT = (
    Path(_ENV_REPO_ROOT).expanduser().resolve()
    if _ENV_REPO_ROOT
    else Path(__file__).resolve().parents[3]
)
if not (_CANDIDATE_ROOT / "src" / "molgen3D").exists():
    cwd = Path.cwd().resolve()
    if (cwd / "src" / "molgen3D").exists():
        _CANDIDATE_ROOT = cwd
REPO_ROOT = _CANDIDATE_ROOT

# Keys that should use geom_data_root instead of data_root
GEOM_DATA_KEYS = {
    "rdkit_folder",
    "test_mols",
    "drugs_summary",
    "conformers_train",
    "conformers_valid",
    "conformers_test",
    "pretokenized_prompts",
    "validation_pickle",

}


def _path_candidate_roots() -> list[Path]:
    """Return ordered roots used when resolving relative paths."""
    roots: list[Path] = []

    def _add_root(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in roots:
            roots.append(resolved)

    if _ENV_PROJECT_ROOT:
        _add_root(Path(_ENV_PROJECT_ROOT).expanduser())

    _add_root(REPO_ROOT)
    for ancestor in REPO_ROOT.parents[:2]:
        _add_root(ancestor)

    return roots


def _absolute_path_candidates(value: str | Path) -> list[Path]:
    """Return the absolute paths to try for a single candidate."""
    candidate = Path(value)
    if candidate.is_absolute():
        return [candidate]

    resolved: list[Path] = []
    seen: set[Path] = set()
    for root in _path_candidate_roots():
        path = (root / candidate).resolve()
        if path in seen:
            continue
        seen.add(path)
        resolved.append(path)

    if not resolved:
        return [candidate]
    return resolved


def _as_path_candidates(value: str | Path | Sequence[str | Path]) -> list[str | Path]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Path)):
        return list(value)
    return [value]


def _resolve_path_value(value: str | Path | Sequence[str | Path]) -> Path:
    """
    Resolve a config path value that may include fallback candidates.
    The first existing candidate is returned; otherwise, the first candidate.
    """
    candidates = _as_path_candidates(value)
    if not candidates:
        raise ValueError("Cannot resolve an empty set of path candidates")

    resolved: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        for candidate_path in _absolute_path_candidates(candidate):
            if candidate_path in seen:
                continue
            seen.add(candidate_path)
            resolved.append(candidate_path)
            if candidate_path.exists():
                return candidate_path

    return resolved[0]


@lru_cache(maxsize=1)
def _cfg() -> dict:
    """Load and cache the paths.yaml configuration file."""
    repo_path = REPO_ROOT / "src" / "molgen3D" / "config" / "paths.yaml"
    if repo_path.exists():
        with repo_path.open("r") as f:
            return yaml.safe_load(f) or {}

    paths_file = pkg_resources.files("molgen3D.config").joinpath("paths.yaml")
    with paths_file.open("r") as f:
        return yaml.safe_load(f) or {}


def _get_config_section(section: str) -> dict:
    """Get a section from the config, returning an empty dict if missing."""
    return _cfg().get(section, {})


def _get_ckpt_base_path(root_rel: str, base_paths: dict) -> Path:
    """Determine the base path for a checkpoint based on root_rel pattern."""

    def _resolve_from_keys(*keys: str, default: str = ".") -> Path:
        for key in keys:
            if key in base_paths:
                return _resolve_path_value(base_paths[key])
        return _resolve_path_value(default)

    if root_rel.startswith("qwen3_06b"):
        return _resolve_from_keys("qwen_yerevann_root", "hf_yerevann_root")
    if "qwen3" in root_rel:
        return _resolve_from_keys("qwen3_grpo_root", "grpo_root")
    if "code_snapshot" in root_rel or "grpo_outputs" in root_rel:
        return _resolve_from_keys("grpo_outputs_root")
    if root_rel.startswith("2025-"):
        return _resolve_from_keys("grpo_root", "ckpts_root")
    return _resolve_from_keys("hf_yerevann_root", default=".")


def _resolve_direct_path(value: str | Path) -> Path:
    """Resolve a single, possibly relative path without a section tag."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return _absolute_path_candidates(candidate)[0]


def load_paths_yaml() -> dict:
    """
    Return a deep copy of the parsed paths.yaml so callers can inspect sections
    without risking shared-state mutations.
    """
    return copy.deepcopy(_cfg())


_DATA_ROOT_KEYS = (
    "ckpts_root",
    "grpo_root",
    "qwen3_grpo_root",
    "hf_yerevann_root",
    "qwen_yerevann_root",
    "geom_data_root",
    "data_root",
    "project_root",
)


def _normalize_data_root(candidate: str | Path) -> Path:
    """Return an absolute resolved data root."""
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=False)


def _collect_data_roots() -> list[Path]:
    """Collect ordered roots to try when resolving data paths."""
    base_paths = load_paths_yaml().get("base_paths", {})
    roots: list[Path] = []
    seen: set[Path] = set()

    for key in _DATA_ROOT_KEYS:
        candidates = _as_path_candidates(base_paths.get(key))
        for candidate in candidates:
            path = Path(candidate)
            normalized = _normalize_data_root(path)
            if normalized not in seen:
                seen.add(normalized)
                roots.append(normalized)

    if not roots:
        roots.append(REPO_ROOT.resolve(strict=False))

    return roots


def _resolve_relative_data_path(relative: Path, roots: list[Path]) -> Path | None:
    for root in roots:
        for variant in _relative_variants(relative):
            candidate = (root / variant).resolve(strict=False)
            if candidate.exists():
                return candidate
    return None


def _resolve_absolute_data_path(absolute: Path, roots: list[Path]) -> Path | None:
    for root in roots:
        try:
            rel = absolute.relative_to(root)
        except ValueError:
            continue
        for fallback in roots:
            for variant in _relative_variants(rel):
                candidate = (fallback / variant).resolve(strict=False)
                if candidate.exists():
                    return candidate
    return None


def _relative_variants(relative: Path) -> list[Path]:
    variants = [relative]
    if relative.parts and relative.parts[0] == "geom_processed":
        remainder = Path(*relative.parts[1:])
        if remainder not in variants:
            variants.append(remainder)
    if relative.parts and relative.parts[0] == "3DMolGen_data":
        remainder = Path("data", *relative.parts[1:])
        if remainder not in variants:
            variants.append(remainder)
    if relative.parts and relative.parts[0] == "data":
        remainder = Path(*relative.parts[1:])
        if remainder not in variants:
            variants.append(remainder)
    if not (relative.parts and relative.parts[0] == "geom_processed"):
        prefixed = Path("geom_processed") / relative
        if prefixed not in variants:
            variants.append(prefixed)
    if not (relative.parts and relative.parts[0] == "3DMolGen_data"):
        prefixed = Path("3DMolGen_data") / relative
        if prefixed not in variants:
            variants.append(prefixed)
    if not (relative.parts and relative.parts[0] == "data"):
        prefixed = Path("data") / relative
        if prefixed not in variants:
            variants.append(prefixed)
    return variants


def resolve_data_path(value: str | Path) -> Path:
    """
    Resolve a data file path against configured base roots.
    """
    candidate = Path(value).expanduser()
    if candidate.exists():
        return candidate.resolve(strict=False)

    roots = _collect_data_roots()
    if candidate.is_absolute():
        resolved = _resolve_absolute_data_path(candidate.resolve(strict=False), roots)
        if resolved:
            return resolved
    else:
        resolved = _resolve_relative_data_path(candidate, roots)
        if resolved:
            return resolved

    return candidate.resolve(strict=False)


def get_ckpt(alias: str, key: str | None = None) -> Path:
    """
    Get the path to a checkpoint for a given model alias and step key.
    
    Args:
        alias: Model alias from the config
        key: Step key (e.g., "1e", "final"). If None, uses "final" if available,
             otherwise the last step alphabetically.
    
    Returns:
        Absolute path to the checkpoint directory
    """
    models = _get_config_section("models")
    entry = models.get(alias)
    if entry is None:
        raise KeyError(f"Unknown model alias '{alias}'.")

    steps = entry.get("steps") or {}
    if not steps:
        raise KeyError(f"Model '{alias}' has no steps defined.")

    if key is None:
        key = "final" if "final" in steps else sorted(steps.keys())[-1]
    if key not in steps:
        raise KeyError(
            f"Step '{key}' not found for '{alias}', "
            f"available: {sorted(steps.keys())}"
        )

    root_rel = entry["root"]
    step_rel = steps[key]
    base_paths = _get_config_section("base_paths")
    base = _get_ckpt_base_path(root_rel, base_paths)

    return base / root_rel / step_rel


def get_tokenizer_path(name: str) -> Path:
    """
    Get the path to a tokenizer by name.
    
    Args:
        name: Tokenizer name from the config
    
    Returns:
        Absolute path to the tokenizer directory
    """
    tokenizers = _get_config_section("tokenizers")
    if name not in tokenizers:
        raise KeyError(f"Unknown tokenizer '{name}', available: {sorted(tokenizers.keys())}")
    return _resolve_path_value(tokenizers[name])


def get_base_path(key: str) -> Path:
    """
    Get a base path by key.
    
    Args:
        key: Base path key from the config
    
    Returns:
        Absolute path
    """
    base_paths = _get_config_section("base_paths")
    if key not in base_paths:
        raise KeyError(f"Unknown base path '{key}', available: {sorted(base_paths.keys())}")
    return _resolve_path_value(base_paths[key])


def get_data_path(key: str) -> Path:
    """
    Get a data path by key.
    
    Args:
        key: Data path key from the config
    
    Returns:
        Absolute path to the data file or directory
    """
    data_cfg = _get_config_section("data")
    if key not in data_cfg:
        raise KeyError(f"Unknown data path '{key}', available: {sorted(data_cfg.keys())}")
    
    rel_candidates = _as_path_candidates(data_cfg[key])
    if not rel_candidates:
        raise ValueError(f"No data path candidates defined for '{key}'")

    base_paths = _get_config_section("base_paths")

    def _base_candidate_values(base_key: str) -> list[str | Path]:
        value = base_paths.get(base_key)
        if value is not None:
            return _as_path_candidates(value)
        if base_key == "geom_data_root":
            return _base_candidate_values("data_root")
        return ["."]

    default_path: Path | None = None
    for rel_candidate in rel_candidates:
        rel_path = Path(rel_candidate)
        if rel_path.is_absolute():
            if default_path is None:
                default_path = rel_path
            if rel_path.exists():
                return rel_path
            continue

        rel_str = str(rel_candidate)
        base_key = (
            "geom_data_root"
            if key in GEOM_DATA_KEYS or rel_str.startswith(("geom_processed", "rdkit_folder"))
            else "data_root"
        )

        for base_value in _base_candidate_values(base_key):
            for base_path in _absolute_path_candidates(base_value):
                candidate_path = base_path / rel_path
                if default_path is None:
                    default_path = candidate_path
                if candidate_path.exists():
                    return candidate_path

    if default_path is not None:
        return default_path

    return Path(rel_candidates[0])


def get_root_path(base_key: str, folder: str | Path) -> Path:
    """
    Return the path under the provided base key for the given folder.
    
    Args:
        base_key: Base path key from the config
        folder: Folder name or path (if absolute, returned as-is)
    
    Returns:
        Absolute path
    """
    folder_path = Path(folder)
    if folder_path.is_absolute():
        return folder_path

    base = get_base_path(base_key)
    return base / folder_path


def get_pretrain_dump_path(folder: str | Path, *, base_key: str = "pretrain_results_root") -> Path:
    """
    Return the path under `base_key` for the provided dump folder.
    
    Args:
        folder: Folder name or path
        base_key: Base path key (default: "pretrain_results_root")
    
    Returns:
        Absolute path
    """
    return get_root_path(base_key, folder)


def get_pretrain_logs_path(folder: str | Path) -> Path:
    """
    Get the path for pretraining logs.
    
    Args:
        folder: Folder name or path
    
    Returns:
        Absolute path
    """
    return get_root_path("pretrain_logs_root", folder)


def get_wandb_path(folder: str | Path) -> Path:
    """
    Get the path for wandb logs.
    
    Args:
        folder: Folder name or path
    
    Returns:
        Absolute path
    """
    return get_root_path("wandb_root", folder)


def get_ckpt_tag_path(key: str) -> Path:
    """
    Resolve a checkpoint alias defined under the `ckpts` section of paths.yaml.

    The key format is `<alias>` or `<alias>/<subpath>`, where `alias` maps to an
    absolute (or repo-relative) directory. Any trailing subpath is appended to
    that base.
    """
    ckpt_cfg = _get_config_section("ckpts")
    if not ckpt_cfg:
        raise KeyError("No 'ckpts' section defined in paths.yaml.")

    normalized = key.strip()
    if not normalized:
        raise KeyError("Empty ckpts key cannot be resolved.")

    alias, sep, remainder = normalized.partition("/")
    alias = alias.strip()
    base = ckpt_cfg.get(alias)
    if base is None:
        raise KeyError(
            f"Unknown ckpts alias '{alias}', available: {sorted(ckpt_cfg.keys())}"
        )
    base_path = _resolve_path_value(base)
    return base_path / remainder if sep else base_path


def resolve_tag(tag: str) -> Path:
    """
    Resolve a structured tag like "base_paths:ckpts_root" into an absolute path.
    
    Supported sections: base_paths, data, tokenizers, ckpts.
    If no colon is present, treats the tag as a direct path.
    
    Args:
        tag: Tag string in format "section:key" or a direct path
    
    Returns:
        Absolute path
    """
    if not tag:
        raise ValueError("Empty tag cannot be resolved")

    if ":" not in tag:
        candidate = Path(tag)
        return candidate if candidate.is_absolute() else _resolve_direct_path(candidate)

    section, key = tag.split(":", 1)
    section = section.strip()
    key = key.strip()

    section_handlers = {
        "base_paths": get_base_path,
        "data": get_data_path,
        "tokenizers": get_tokenizer_path,
        "ckpts": get_ckpt_tag_path,
    }

    handler = section_handlers.get(section)
    if handler is None:
        raise KeyError(
            f"Unsupported tag section '{section}' in '{tag}'. "
            f"Expected one of: {', '.join(section_handlers.keys())}."
        )

    return handler(key)
