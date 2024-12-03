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
        self.model1.eval()
        self.model2.eval()
        # Ensure both models are on the same device
        self.device = next(self.model1.parameters()).device
        self.model2.to(self.device)

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
      """
      Override prediction_step to ensure ensemble prediction during evaluation.
      """
      # Move inputs to the device
      inputs = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
      model_inputs = {key: value for key, value in inputs.items() if key in {"input_ids", "attention_mask", "token_type_ids"}}
      
      # Set models to evaluation mode
      self.model1.eval()
      self.model2.eval()
      
      # Compute outputs for both models
      with torch.no_grad():
          outputs1 = self.model1(**model_inputs)
          outputs2 = self.model2(**model_inputs)

      # Average the logits
      mean_logits = self.w1 * outputs1.logits + self.w2 * outputs2.logits

      # Ensure logits have the correct dimensions (batch_size, num_classes)
      if mean_logits.ndimension() == 1:
          mean_logits = mean_logits.unsqueeze(0)

      # Compute loss if required
      loss = None
      labels = inputs.get("labels", None)
      if labels is not None:
          labels = labels.to(self.device)
          if not prediction_loss_only:
              loss = torch.nn.functional.cross_entropy(mean_logits, labels)

      # Return the loss, logits (predictions), and labels (all as tensors)
      return (loss, mean_logits, labels)
