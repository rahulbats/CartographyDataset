import hashlib
from transformers import TrainerCallback
import torch
import numpy as np
import os
import json

class CartographyCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.training_dynamics = {
            "example_ids": [],
            "confidence": {},
            "variability": {},
            "correctness": {},
        }
    
    def on_train_begin(self, args, state, control, **kwargs):
        print("on_train_begin invoked")  # Add a print statement to see if the training is starting

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Capture training dynamics during the logging phase.
        """
        
        inputs = getattr(state, "inputs", None)
        outputs = getattr(state, "outputs", None)
        if outputs is None or inputs is None:
            return  # If data is not available, skip

        logits = outputs.get("logits")
        labels = inputs.get("labels")
        example_ids = getattr(state, "example_ids", None)
        

        if logits is None or labels is None :
            return  # If required data is missing, skip

        # Compute predictions and probabilities
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        max_probs = np.max(probabilities, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        correct = (predictions == labels.cpu().numpy())

        # Log dynamics
        for i in range(len(labels)):
            # Generate a unique hash for example_id
            example_id = example_ids[i]

            if example_id not in self.training_dynamics["example_ids"]:
                self.training_dynamics["example_ids"].append(example_id)
                self.training_dynamics["confidence"][example_id] = []
                self.training_dynamics["variability"][example_id] = []
                self.training_dynamics["correctness"][example_id] = []

            self.training_dynamics["confidence"][example_id].append(float(max_probs[i]))
            self.training_dynamics["variability"][example_id].append(probabilities[i].tolist())
            self.training_dynamics["correctness"][example_id].append(int(correct[i]))

    def on_save(self, args, state, control, **kwargs):
        # Save the training dynamics to the checkpoint directory
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not checkpoint_dir:
            print("No checkpoint directory inferred. Skipping saving training dynamics.")
            return

        os.makedirs(checkpoint_dir, exist_ok=True)
        json_path = os.path.join(checkpoint_dir, "training_dynamics.json")
        with open(json_path, "w") as f:
            json.dump(self.training_dynamics, f)
        print(f"Training dynamics saved to {json_path}")
