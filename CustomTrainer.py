class CustomTrainer(Trainer):
    def __init__(self, *args, gamma=2, alpha=0.25, **kwargs):
        """
        Extend Trainer to support Focal Loss parameters.
        gamma: Focusing parameter for Focal Loss.
        alpha: Balancing parameter for Focal Loss.
        """
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha)  # Initialize Focal Loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to use Focal Loss.
        """
        labels = inputs["labels"]
        example_ids = inputs.pop("example_ids", None)
        outputs = model(**inputs)
        logits = outputs.logits

        # Compute Focal Loss
        loss = self.focal_loss(logits, labels)

        # Attach inputs and outputs to the trainer's state for the callback
        self.state.inputs = inputs
        self.state.outputs = outputs
        self.state.example_ids = example_ids

        return (loss, outputs) if return_outputs else loss
