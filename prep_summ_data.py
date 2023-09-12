import json
import pickle
import os
import argparse

def restructure_dataset(file_name):
    # Generate dataset name and output file name based on the input file name
    dataset_name = os.path.splitext(file_name)[0]  # Remove the file extension to get the base name
    output_file_name = f"/data/TrainLLMv3/source/{dataset_name}.pkl"
    print(output_file_name)

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
        summary = value.get('summary', [])
        restructured_dataset['train'].append({
            'context': context,
            'summary': summary
        })

    # Step 5: Update the datasets_ object
    datasets_[dataset_name] = restructured_dataset

    # Step 7: Save the datasets_ object to disk as a pickle file
    with open(output_file_name, "wb") as f:
        pickle.dump(datasets_, f)
        
def main():
    parser = argparse.ArgumentParser(description='Restructure dataset.')
    parser.add_argument('file_name', type=str, help='Input JSON file name to process')
    args = parser.parse_args()

    restructure_dataset(args.file_name)

if __name__ == "__main__":
    main()