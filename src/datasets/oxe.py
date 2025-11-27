from typing import List, Dict, Any
from pathlib import Path
import json

import numpy as np
import tensorflow as tf

from .base import BaseDataset


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
            dataset_name_2/
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
        if not data_path.exists():
            return []

        episode_names = []

        # Find all dataset directories
        for dataset_dir in sorted(data_path.iterdir()):
            if not dataset_dir.is_dir():
                continue

            # Look for version subdirectory (e.g., 0.1.0)
            version_dirs = list(dataset_dir.glob("*.*.*"))
            if not version_dirs:
                continue

            version_dir = version_dirs[0]  # Use first version found
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
        dataset_name, shard_id, episode_in_shard = parts[0], parts[1], int(parts[2])
        shard_idx = int(shard_id)

        # Get cached dataset info
        if dataset_name not in self._dataset_cache:
            raise ValueError(f"Dataset {dataset_name} not found in cache")

        dataset_info = self._dataset_cache[dataset_name]['info']
        version_dir = self._dataset_cache[dataset_name]['version_dir']

        # Get split info
        split_info = dataset_info['splits'][0]  # Assume first split (train)
        shard_lengths = [int(length) for length in split_info['shardLengths']]
        num_shards = len(shard_lengths)

        # Construct TFRecord file path
        file_template = split_info['filepathTemplate']
        filename = file_template.format(
            DATASET=dataset_name,
            SPLIT=split_info['name'],
            FILEFORMAT=dataset_info['fileFormat'],
            SHARD_X_OF_Y=f"{shard_idx:05d}-of-{num_shards:05d}"
        )
        tfrecord_path = version_dir / filename

        # Parse episode from TFRecord
        episode_data = self._parse_tfrecord_episode(tfrecord_path, episode_in_shard)

        # Add video_name to the returned data
        episode_data['video_name'] = video_name

        # Add TFRecord metadata for tracking
        episode_data['metadata']['tfrecord_info'] = {
            'dataset_name': dataset_name,
            'shard_idx': shard_idx,
            'episode_in_shard': episode_in_shard,
            'split': split_info['name'],
            'tfrecord_path': str(tfrecord_path.relative_to(self.data_dir))
        }

        return episode_data

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
        """Extract structured data from parsed TFRecord example.

        Args:
            example: Parsed TFRecord example

        Returns:
            Dict with frames, description, and metadata
        """
        features = example.features.feature

        # The structure is: steps/field_name where each field contains all timesteps
        # Extract images (each is a PNG/JPEG encoded byte string)
        frames = []
        if 'steps/observation/image' in features:
            image_bytes_list = features['steps/observation/image'].bytes_list.value
            for image_bytes in image_bytes_list:
                # Decode PNG/JPEG image
                image = tf.image.decode_image(image_bytes, channels=3)
                frames.append(image.numpy())

        # Extract states (flattened array of all timesteps)
        states = []
        if 'steps/observation/state' in features:
            state_values = list(features['steps/observation/state'].float_list.value)
            # Figure out state dimension from number of frames and total values
            num_steps = len(frames)
            if num_steps > 0 and len(state_values) > 0:
                state_dim = len(state_values) // num_steps
                # Reshape into (num_steps, state_dim)
                states = np.array(state_values).reshape(num_steps, state_dim)

        # Extract actions (flattened array of all timesteps)
        actions = []
        if 'steps/action' in features:
            action_values = list(features['steps/action'].float_list.value)
            num_steps = len(frames)
            if num_steps > 0 and len(action_values) > 0:
                action_dim = len(action_values) // num_steps
                # Reshape into (num_steps, action_dim)
                actions = np.array(action_values).reshape(num_steps, action_dim)

        # Extract language instruction (first one, they're all the same)
        language_instruction = ""
        if 'steps/language_instruction' in features:
            lang_bytes_list = features['steps/language_instruction'].bytes_list.value
            if len(lang_bytes_list) > 0:
                language_instruction = lang_bytes_list[0].decode('utf-8')

        # Extract language embedding (flattened, should be 512-dim)
        language_embedding = None
        if 'steps/language_embedding' in features:
            emb_values = list(features['steps/language_embedding'].float_list.value)
            num_steps = len(frames)
            if num_steps > 0 and len(emb_values) > 0:
                emb_dim = len(emb_values) // num_steps
                # Just take the first one since they're all the same
                language_embedding = np.array(emb_values[:emb_dim])

        # Convert lists to numpy arrays
        frames = np.array(frames) if frames else np.array([])
        states = np.array(states) if len(states) > 0 else np.array([])
        actions = np.array(actions) if len(actions) > 0 else np.array([])

        # Build metadata dict
        metadata = {}
        if len(states) > 0:
            metadata['state'] = states
        if len(actions) > 0:
            metadata['action'] = actions
        if language_embedding is not None:
            metadata['language_embedding'] = language_embedding

        return {
            'frames': frames,
            'description': language_instruction,
            'metadata': metadata
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
        print(f"  description: {data_dict['description']}")
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
