import json
import pickle
import os

def restructure_dataset(file_name):
    # Generate dataset name and output file name based on the input file name
    dataset_name = os.path.splitext(file_name)[0]  # Remove the file extension to get the base name
    output_file_name = f"./source/{dataset_name}.pkl"

    # Step 1: Load the JSON file
    with open(file_name, "r") as f:
        my_dataset = json.load(f)

    # Step 2: Prepare the datasets_ object
    datasets_ = {}
    datasets_[dataset_name] = my_dataset

    # Step 3: Extract the existing dataset
    existing_dataset = datasets_[dataset_name]

    # Step 4: Restructure the dataset
    restructured_dataset = {'train': []}
    for key, value in existing_dataset.items():
        context = value.get('context', '')
        qa_pairs = value.get('qa_pairs', [])
        for qa_pair in qa_pairs:
            question = qa_pair.get('question', '')
            answer = qa_pair.get('answer', '')
            restructured_dataset['train'].append({
                'context': context,
                'question': question,
                'answer': answer
            })

    # Step 5: Update the datasets_ object
    datasets_[dataset_name] = restructured_dataset

    # Step 6: Add indices to records
    train_partition = restructured_dataset['train']
    train_indices = list(range(len(train_partition)))

    def add_indices_to_records(partition, indices):
        for i, record in enumerate(partition):
            record['index'] = indices[i]

    add_indices_to_records(train_partition, train_indices)

    # Step 7: Save the datasets_ object to disk as a pickle file
    with open(output_file_name, "wb") as f:
        pickle.dump(datasets_, f)