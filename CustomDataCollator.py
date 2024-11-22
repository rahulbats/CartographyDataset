from transformers import DataCollatorWithPadding, DefaultDataCollator

class DataCollatorWithIds(DefaultDataCollator):
    def __call__(self, features):
        # Extract `example_id` from features
        example_ids = [feature.pop("example_ids", None) for feature in features]

        # Call the parent collator for padding
        batch = super().__call__(features)

        # Add `example_id` back to the batch
        batch["example_ids"] = example_ids
        return batch
