import os
import json
import torch
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler, random_split, Dataset
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import hydra
import random
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.datasets import (GroupedTaskDataset, SequenceVLDataset)
from libero.lifelong.utils import (get_task_embs, safe_device, create_experiment_dir)

from mode.datasets.utils.libero_utils import get_dataset, get_split_dataset

class TranslatedSequenceVLDataset(Dataset):
    def __init__(
        self,
        sequence_dataset,
        task_emb,
        task_description,
        plan_text=None,
        obs_seq_len: int =1,
        act_seq_len: int =1,
        transforms=None
    ):
        self.obs_seq_len = obs_seq_len
        self.act_seq_len = act_seq_len
        self.transforms = hydra.utils.instantiate(transforms)
        self.sequence_dataset = sequence_dataset
        # add goal mode to the sequence dataset
        self.sequence_dataset.goal_mode = "last"
        self.task_emb = task_emb
        self.task_description = task_description
        self.plan_text = plan_text if plan_text is not None else task_description
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        main_dict = {}
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_emb"] = self.task_emb
        return_dict["lang_text"] = self.task_description
        return_dict["plan_text"] = self.plan_text
        return_dict = self.get_des_act_obs_sequence(return_dict)
        main_dict["lang"] = self.translation_dict(return_dict)
        # main_dict['modality'] = 'lang'

        # Apply transforms
        if self.transforms:
            main_dict = self.apply_transforms(main_dict["lang"])
        main_dict['idx'] = idx
        return main_dict

    def apply_transforms(self, data, train=True):
        # Assuming data contains images in 'rgb_static' and 'rgb_gripper'
        if train:
            transforms = self.transforms['train']
        for key in data['rgb_obs']:
            x = data['rgb_obs'][key]
            x = torch.from_numpy(x).byte().permute(0, 3, 1, 2)
            for transform in transforms[key]:
                x = transform(x)
            data['rgb_obs'][key] = x
            # data['rgb_obs'][key] = transforms[key](data['rgb_obs'][key])

        return data

    def get_des_act_obs_sequence(self, return_dict):

        for key in return_dict['obs']:
            return_dict['obs'][key] = return_dict['obs'][key][:self.obs_seq_len]
        return_dict['actions'] = return_dict['actions'][:self.act_seq_len]

        return_dict['robot_obs'] = return_dict['obs']['joint_states'][:self.obs_seq_len]
        return_dict['gripper_states'] = return_dict['obs']['gripper_states'][:self.obs_seq_len]

        return return_dict

    def translation_dict(self, dict):
        translated_dict = {}
        # dict['obs'] = self.combine_goal_obs_with_obs(dict['obs'], dict['goal_obs'])
        if 'obs' in dict.keys():
            translated_dict['rgb_obs'] = {}
            translated_dict["rgb_obs"]['rgb_static'] = dict['obs']['agentview_rgb']
            translated_dict["rgb_obs"]['rgb_gripper'] = dict['obs']['eye_in_hand_rgb']
            translated_dict['robot_obs'] = dict['obs']['joint_states']
            # translated_dict['gripper_states'] = dict['obs']['gripper_states']

        translated_dict['lang_text'] = dict['lang_text']
        translated_dict['plan_text'] = dict['plan_text']
        translated_dict['depth_obs'] = {}
        translated_dict['actions'] = dict['actions']
        # translated_dict['robot_obs'] = dict['robot_obs']
        translated_dict['robot_obs'] = np.concatenate([dict['robot_obs'], np.expand_dims(dict['obs']['gripper_states'][0], 0)], axis=-1)
        return translated_dict

    def combine_goal_obs_with_obs(self, obs, goal_obs):
        combined_obs = {}
        for key in obs:
            if key in ('actions'):
                combined_obs[key] = obs[key]
            else:
                combined_obs[key] = np.concatenate([obs[key], np.expand_dims(goal_obs[key], axis=0)],axis=0)
        return combined_obs


class LiberoDataModule(pl.LightningDataModule):

    def __init__(
        self,
        datasets: DictConfig,
        observation_space:  DictConfig,
        num_workers: int = 8,
        transforms: DictConfig = None,  # Replace with your default transforms
        shuffle_val: bool = False,
        benchmark_name: str = 'libero_goal',
        task_embedding_format: str = 'clip',
        split_ratio: float = 0.0,
        plan_file: str = "",
        **kwargs,
    ):
        super().__init__()
        self.datasets_cfg = datasets
        self.num_workers = num_workers
        self.transforms = transforms
        self.shuffle_val = shuffle_val
        self.train_datasets = []
        self.val_datasets = []
        self.train_sampler = None
        self.val_sampler = None
        self.modalities = []
        self.benchmark_name = benchmark_name
        self.task_embedding_format = task_embedding_format
        self.split_ratio = split_ratio
        self.plan_file = plan_file
        self.plan_by_task = {}
        self._load_plan_mapping()

    def _resolve_plan_file(self):
        """Resolve plan_file, tolerating Hydra changing the working directory.

        If the given path is not found (e.g. relative path + Hydra chdir to the
        run dir), fall back to resolving it against the repo root (two levels up
        from this file).
        """
        if os.path.exists(self.plan_file):
            return self.plan_file
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidate = os.path.join(repo_root, os.path.basename(self.plan_file))
        if os.path.exists(candidate):
            return candidate
        return None

    def _load_plan_mapping(self):
        if not self.plan_file:
            return
        resolved = self._resolve_plan_file()
        if resolved is None:
            print(f"[LiberoDataModule] plan_file not found: {self.plan_file}")
            return
        self.plan_file = resolved

        with open(self.plan_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                task = obj.get("task", "")
                plan_text = obj.get("plan_text", "")
                if task and plan_text:
                    self.plan_by_task[task] = plan_text
        print(f"[LiberoDataModule] loaded plan_text for {len(self.plan_by_task)} tasks")

    def create_cfg_for_libero(self, task_embedding_format):
        self.cfg = DictConfig({'task_embedding_format': task_embedding_format,
                               'data': {'max_word_len': 25}})

        self.cfg.policy = OmegaConf.create()
        self.cfg.policy.language_encoder = OmegaConf.create()
        self.cfg.policy.language_encoder.network_kwargs = OmegaConf.create()

    def translate_obs_space(self, obs_space):
        self.libero_observation_space = {}
        self.libero_observation_space['rgb'] = obs_space['rgb_obs']

        # self.libero_observation_space['modality']['depth'] = []
        self.libero_observation_space['low_dim'] = obs_space['state_obs']

    def _parse_benchmark_names(self):
        """Allow benchmark_name to be a single name or a comma-separated list.

        For "全集" training we point this at the four base suites
        (libero_spatial, libero_object, libero_goal, libero_10) which together
        make up the 40 tasks covered by the plan file.
        """
        raw = self.benchmark_name
        if isinstance(raw, (list, tuple)):
            names = list(raw)
        else:
            names = [n.strip() for n in str(raw).split(",")]
        return [n for n in names if n]

    def _initialize_datasets(self, datasets_cfg):
        self.translate_obs_space(datasets_cfg.lang_dataset.obs_space)

        benchmark_names = self._parse_benchmark_names()
        datasets_default_path = datasets_cfg.get('custom_data_path', get_libero_path("datasets"))

        train_datasets = []
        val_datasets = []

        # Global task counter so obs-utils is initialized exactly once, on the
        # very first task across all benchmarks.
        global_task_idx = 0

        for benchmark_name in benchmark_names:
            benchmark_instance = get_benchmark(benchmark_name)()
            num_tasks = benchmark_instance.get_num_tasks()
            descriptions = []

            for i in range(num_tasks):
                dataset_path = os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(i))
                task_name = os.path.splitext(os.path.basename(dataset_path))[0]

                task_i_dataset, shape_meta = get_dataset(
                    dataset_path=dataset_path,
                    obs_modality=self.libero_observation_space,
                    initialize_obs_utils=(global_task_idx == 0),
                    seq_len=datasets_cfg.lang_dataset.action_seq_len,
                )
                global_task_idx += 1
                descriptions.append(benchmark_instance.get_task(i).language)

                task_embs = get_task_embs(self.cfg, descriptions)
                benchmark_instance.set_task_embs(task_embs)
                vl_dataset = TranslatedSequenceVLDataset(
                    task_i_dataset,
                    task_embs[i],
                    descriptions[i],
                    plan_text=self.plan_by_task.get(task_name, descriptions[i]),
                    act_seq_len=datasets_cfg.lang_dataset.action_seq_len,
                    obs_seq_len=datasets_cfg.lang_dataset.obs_seq_len,
                    transforms=self.transforms
                )

                train_datasets.append(vl_dataset)

        print(f"[LiberoDataModule] built {len(train_datasets)} task datasets "
              f"from benchmarks: {benchmark_names}")

        # Concatenate all training and validation datasets
        # concat_img_datasets = ConcatDataset(img_datasets)
        concat_train_datasets = ConcatDataset(train_datasets)

        val_datasets = {
            'lang': concat_train_datasets,
        }

        datasets = {
            'lang': concat_train_datasets,
        }

        return datasets, val_datasets

    def setup(self, stage=None):

        # Initialize datasets
        self.create_cfg_for_libero(self.task_embedding_format)
        self.train_datasets, self.val_datasets = self._initialize_datasets(self.datasets_cfg)
        self.modalities.append('lang')


    def train_dataloader(self):
        return {
            key: DataLoader(
                dataset,
                batch_size=self.datasets_cfg.lang_dataset.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=True,
            )
            for key, dataset in self.train_datasets.items()
        }

    def val_dataloader(self):
        return {
            key: DataLoader(
                dataset,
                batch_size=self.datasets_cfg.lang_dataset.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
            ) for key, dataset in self.val_datasets.items()
        }


def split_trajectories(num_trajectories, train_ratio=0.8):
    num_train = int(num_trajectories * train_ratio)
    num_val = num_trajectories - num_train
    trajectory_indices = list(range(num_trajectories))
    random.shuffle(trajectory_indices)

    train_indices = trajectory_indices[:num_train]
    val_indices = trajectory_indices[num_train:]

    return train_indices, val_indices
