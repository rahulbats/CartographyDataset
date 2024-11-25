import base64
from generate_data_map import get_focussed_sets

class DatasetFilter:
    def __init__(self, checkpoint_dir, max_confidence=0.9, max_variability=None, max_correctness=0.5):
       
        self.hashes = get_focussed_sets(checkpoint_dir,max_confidence, max_variability, max_correctness)


    def filter_function(self, example):
        """
        Define the filtering condition for the dataset.
        :param example: A single example from the dataset.
        :return: True if the example satisfies the condition, False otherwise.
        """
        return self.hashes.__contains__(example["example_id"])
    
