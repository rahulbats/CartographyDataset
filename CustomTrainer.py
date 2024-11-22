from transformers import Trainer
import torch

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to capture logits and labels.
        """
        labels = inputs["labels"]
        example_ids = inputs.pop("example_ids", None)
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(logits, labels)

        # Attach inputs and outputs to the trainer's state for the callback
        self.state.inputs = inputs
        self.state.outputs = outputs
        self.state.example_ids = example_ids

        return (loss, outputs) if return_outputs else loss
