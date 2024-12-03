from logging import exception
from transformers import Trainer
import torch
import numpy as np

class TwoModelMeanTrainer(Trainer):
    def __init__(self, model1, model2,w1, w2, *args, **kwargs):
        """
        Extend Trainer to support two models whose outputs are averaged.
        """
        super().__init__(*args, **kwargs)
        self.model1 = model1
        self.model2 = model2
        if (w1+w2)!= 1:
          raise Exception("weights have to add up to 1")
        self.w1=w1
        self.w2=w2

        # Ensure both models are on the same device
        self.device = next(self.model1.parameters()).device
        self.model2.to(self.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to use the mean logits from two models.
        """
        # Move inputs to the device
        inputs = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
        # Remove unnecessary fields (e.g., example_ids)
        model_inputs = {key: value for key, value in inputs.items() if key in {"input_ids", "attention_mask", "token_type_ids", "labels"}}


        # Compute outputs for both models
        outputs1 = self.model1(**model_inputs)
        
        outputs2 = self.model2(**model_inputs)
        #print(f"ouput1 {outputs1.logits}")
        #print(f"ouput2 {outputs2.logits}")
        # Average the logits
        mean_logits = self.w1*outputs1.logits + self.w2*outputs2.logits

        # Compute loss using Cross Entropy
        labels = inputs["labels"]
        loss = torch.nn.functional.cross_entropy(mean_logits, labels)

        # Attach inputs and outputs to the trainer's state for callbacks
        self.state.inputs = inputs
        self.state.outputs = outputs1
        return (loss, outputs1) if return_outputs else loss
