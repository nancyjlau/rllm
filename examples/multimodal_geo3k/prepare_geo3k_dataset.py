"""
Prepare the Geometry3K dataset for multimodal VL training.

This dataset contains geometry math problems with accompanying images,
suitable for testing multimodal models like Qwen2.5-VL and Qwen3-VL.
"""

from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry


def prepare_geo3k_data():
    """
    Load and preprocess the Geometry3K dataset.

    The dataset structure from hiyouga/geometry3k:
    - problem: The geometry question text
    - images: List of image URLs/paths
    - answer: The ground truth answer
    """

    # Load from HuggingFace
    dataset = load_dataset("hiyouga/geometry3k")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    def preprocess_fn(example):
        """
        Preprocess to match rLLM's expected format with question and ground_truth.
        Keep images field for multimodal processing.
        """
        return {
            "question": example.get("problem", ""),
            "ground_truth": example.get("answer", ""),
            "images": example.get("images", []),
            "data_source": "geometry3k",
        }

    train_dataset = train_dataset.map(preprocess_fn)
    test_dataset = test_dataset.map(preprocess_fn)

    # Register with DatasetRegistry
    DatasetRegistry.register_dataset("geometry3k", train_dataset, "train")
    DatasetRegistry.register_dataset("geometry3k", test_dataset, "test")

    print(f"Successfully registered geometry3k dataset:")
    print(f"  Train size: {len(train_dataset)}")
    print(f"  Test size: {len(test_dataset)}")
    print(f"\nExample item:")
    print(f"  Question: {train_dataset[0]['question'][:100]}...")
    print(f"  Answer: {train_dataset[0]['ground_truth']}")
    print(f"  Images: {train_dataset[0]['images']}")


if __name__ == "__main__":
    prepare_geo3k_data()
