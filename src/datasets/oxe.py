from typing import List, Dict, Any
from pathlib import Path
import json
import os

import numpy as np
import tensorflow as tf

from .base import BaseDataset

try:
    tf.config.set_visible_devices([], "GPU")
except Exception as e:
    print(f"[WARN] TF set_visible_devices([],'GPU') failed: {e}")

class OXEDataset(BaseDataset):
    """Open X-Embodiment dataset loader for TFRecord format.

    The dataset is organized as:
        data_dir/
            dataset_name_1/
                0.1.0/
                    dataset_name_1-train.tfrecord-00000-of-00008
                    dataset_name_1-train.tfrecord-00001-of-00008
                    ...
                    dataset_info.json
                    features.json
            dataset_name_2/asu_table_top_converted_externally_to_rlds/0.1.0
                ...

    Each episode contains timesteps with:
        - observation: dict with 'image', 'state', etc.
        - action: robot action vector
        - language_instruction: natural language description
        - language_embedding: 512-dim embedding
    """

    def __init__(self, data_dir: str = "vla-dataset-samples/open-x-embodiment"):
        self._dataset_cache = {}  # Cache dataset info to avoid re-reading JSON
        super().__init__(data_dir)

    def get_video_names(self) -> List[str]:
        """Discover all episodes across all datasets.

        Returns:
            List of episode identifiers in format: "{dataset_name}/{shard_id}/{episode_in_shard}"
            Example: "asu_table_top_converted_externally_to_rlds/00000/0"
        """
        data_path = Path(self.data_dir)
        print(f"[INFO] Scanning data directory: {data_path}")
        if not data_path.exists():
            return []

        episode_names = []

        # Find all dataset directories
        for dataset_dir in sorted(data_path.iterdir()):
            print(f"[DEBUG] Checking dataset directory: {dataset_dir}")
            if not dataset_dir.is_dir():
                continue

            # Look for version subdirectory (e.g., 0.1.0)
            version_dirs = list(dataset_dir.glob("*.*.*"))
            if not version_dirs:
                continue

            version_dir = version_dirs[0]  # Use first version found
            print(f"[INFO] Loading dataset: {dataset_dir.name} (version: {version_dir.name})")
            dataset_info_path = version_dir / "dataset_info.json"

            if not dataset_info_path.exists():
                continue

            # Load dataset info to get episode count
            with open(dataset_info_path, 'r') as f:
                dataset_info = json.load(f)

            # Cache the info for later use
            dataset_name = dataset_dir.name
            self._dataset_cache[dataset_name] = {
                'info': dataset_info,
                'version_dir': version_dir
            }

            # Create episode identifiers with shard information
            for split in dataset_info.get('splits', []):
                shard_lengths = split.get('shardLengths', [])

                # Create identifiers for each episode in each shard
                for shard_idx, shard_length in enumerate(shard_lengths):
                    
                    # Temporal safeguard for using example data
                    tfrecord_path = self._get_tfrecord_path(dataset_name, shard_idx)
                    if not os.path.isfile(tfrecord_path):
                        continue

                    shard_length = int(shard_length)
                    for episode_in_shard in range(shard_length):
                        # Format: dataset_name/shard_id/episode_in_shard
                        shard_id = f"{shard_idx:05d}"
                        episode_names.append(f"{dataset_name}/{shard_id}/{episode_in_shard}")

        return episode_names

    def get_video(self, video_name: str) -> Dict[str, Any]:
        """Load a single episode from TFRecord files.

        Args:
            video_name: Episode identifier in format "{dataset_name}/{shard_id}/{episode_in_shard}"
                       Example: "asu_table_top_converted_externally_to_rlds/00000/0"

        Returns:
            Dict containing:
                - video_name: str episode identifier
                - frames: np.ndarray of shape (num_steps, H, W, 3)
                - description: str language instruction
                - metadata: dict with robot state, actions, embeddings, and tfrecord_info
        """
        # Parse video name
        parts = video_name.split("/")
        dataset_name, shard_idx, episode_in_shard = parts[0], int(parts[1]), int(parts[2])

        # Get cached dataset info
        if dataset_name not in self._dataset_cache:
            raise ValueError(f"Dataset {dataset_name} not found in cache")

        # Get TFRecord path
        tfrecord_path = self._get_tfrecord_path(dataset_name, shard_idx)

        # Parse episode from TFRecord
        episode_data = self._parse_tfrecord_episode(tfrecord_path, episode_in_shard)

        # Add video_name to the returned data
        episode_data['video_name'] = video_name

        # Add TFRecord metadata for tracking
        episode_data['metadata']['tfrecord_info'] = {
            'dataset_name': dataset_name,
            'shard_idx': shard_idx,
            'episode_in_shard': episode_in_shard,
            'split': tfrecord_path.name.split('-', 1)[1].split('.', 1)[0],
            'tfrecord_path': str(tfrecord_path.relative_to(self.data_dir))
        }

        return episode_data
    
    def _get_tfrecord_path(self, dataset_name, shard_idx):
        """
        Make TFRecord path robust:
        1) Try dataset_info filepathTemplate (original behavior)
        2) If not exists, fallback to common patterns / glob search in version_dir
        3) Finally, pick shard_idx-th file if multiple exist
        """
        dataset_info = self._dataset_cache[dataset_name]['info']
        version_dir = self._dataset_cache[dataset_name]['version_dir']

        # --------
        # 1) Original: use dataset_info template
        # --------
        try:
            split_info = dataset_info['splits'][0]  # Assume first split
            shard_lengths = [int(length) for length in split_info.get('shardLengths', [])]
            num_shards = len(shard_lengths) if len(shard_lengths) > 0 else None

            file_template = split_info.get('filepathTemplate', None)
            if file_template is not None and num_shards is not None:
                filename = file_template.format(
                    DATASET=dataset_name,
                    SPLIT=split_info.get('name', 'train'),
                    FILEFORMAT=dataset_info.get('fileFormat', 'tfrecord'),
                    SHARD_X_OF_Y=f"{shard_idx:05d}-of-{num_shards:05d}"
                )
                tfrecord_path = version_dir / filename
                if tfrecord_path.is_file():
                    return tfrecord_path
        except Exception:
            pass

        # --------
        # 2) Fallback: glob search (handles "bridge_oxe.tfrecord-00000-of-01024" etc.)
        # --------
        # try split-included and split-not-included patterns
        candidates = sorted(version_dir.glob(f"{dataset_name}*.tfrecord-*"))
        if not candidates:
            # also allow tfrecord without dash formatting (just in case)
            candidates = sorted(version_dir.glob(f"{dataset_name}*.tfrecord*"))

        if not candidates:
            raise FileNotFoundError(
                f"No TFRecord files found for dataset={dataset_name} under {version_dir}"
            )

        # If shard_idx is within available files, use it; otherwise clamp to last
        if shard_idx < len(candidates):
            return candidates[shard_idx]
        return candidates[-1]


    def _parse_tfrecord_episode(self, tfrecord_path: Path, episode_offset: int) -> Dict[str, Any]:
        """Parse a specific episode from a TFRecord file.

        Args:
            tfrecord_path: Path to TFRecord file
            episode_offset: Index of episode within this shard

        Returns:
            Dict with frames, description, and metadata
        """
        # Read the TFRecord file
        dataset = tf.data.TFRecordDataset(str(tfrecord_path))

        # Skip to the target episode
        episode = None
        for idx, record in enumerate(dataset):
            if idx == episode_offset:
                episode = record
                break

        if episode is None:
            raise ValueError(f"Episode at offset {episode_offset} not found")

        # Parse the serialized example
        parsed = tf.train.Example.FromString(episode.numpy())

        # Extract data from the parsed example
        return self._extract_episode_data(parsed)


    def _extract_episode_data(self, example: tf.train.Example) -> Dict[str, Any]:
        features = example.features.feature

        # -------------------------
        # 1) Frames: language_table uses steps/observation/rgb
        #    (fallback: steps/observation/image for other datasets)
        # -------------------------
        frames = []
        if "steps/observation/rgb" in features:
            image_bytes_list = features["steps/observation/rgb"].bytes_list.value
            for b in image_bytes_list:
                img = tf.image.decode_image(b, channels=3)
                frames.append(img.numpy())
        elif "steps/observation/image" in features:
            image_bytes_list = features["steps/observation/image"].bytes_list.value
            for b in image_bytes_list:
                img = tf.image.decode_image(b, channels=3)
                frames.append(img.numpy())

        frames = np.asarray(frames, dtype=np.uint8) if len(frames) > 0 else np.array([], dtype=np.uint8)

        num_steps = len(frames)
        # -------------------------
        # 3) Action (language_table: steps/action shape [2])
        # -------------------------
        actions = np.array([], dtype=np.float32)
        if "steps/action" in features and num_steps > 0:
            action_values = list(features["steps/action"].float_list.value)
            if len(action_values) > 0:
                action_dim = len(action_values) // num_steps
                actions = np.asarray(action_values, dtype=np.float32).reshape(num_steps, action_dim)

        # -------------------------
        # 4) Minimal metadata (필요한 것만)
        # -------------------------
        metadata = {}
        if actions.size > 0:
            metadata["action"] = actions

        # (옵션) language_table의 effector 2D 좌표들 같이 쓰고 싶으면 여기 추가 가능
        # if "steps/observation/effector_translation" in features and num_steps > 0:
        #     v = list(features["steps/observation/effector_translation"].float_list.value)
        #     metadata["effector_translation"] = np.asarray(v, dtype=np.float32).reshape(num_steps, 2)

        return {
            "frames": frames,
            "metadata": metadata,
        }







if __name__ == "__main__":
    dataset = OXEDataset()
    print(f"Data directory: {dataset.data_dir}")
    print(f"Number of episodes: {len(dataset)}")
    print(f"Episode names (first 5): {dataset.video_names[:5]}\n")

    if len(dataset) > 0:
        data_dict = dataset[0]
        print(f"\nFirst episode:")
        print(f"  video_name: {data_dict['video_name']}")
        print(f"  frames shape: {data_dict['frames'].shape}")
        print(f"  descriptions: {data_dict['descriptions']}")
        print(f"  metadata keys: {list(data_dict['metadata'].keys())}")

        for key, value in data_dict['metadata'].items():
            if key == 'tfrecord_info':
                print(f"\n  tfrecord_info:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            elif isinstance(value, np.ndarray):
                print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"  {key}: {value}")
