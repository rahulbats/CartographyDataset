import base64

# This function preprocesses an NLI dataset, tokenizing premises and hypotheses.
def insert_hash(examples):
    
    example_ids = []
    for premise, hypothesis in zip(examples['premise'], examples['hypothesis']):
        stringToEncode = f"{premise}|||{hypothesis}"
        encoded = stringToEncode.encode('utf-8')
        example_id = base64.b64encode(encoded).decode('utf-8')
        example_ids.append(example_id)
        

    examples['example_id'] = example_ids
    return examples
