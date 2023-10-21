import json

# Replace with the path to your actual file
input_file_path = 'results.json'

# Read the JSON file
with open(input_file_path, 'r', encoding='utf-8') as file:
    content = json.load(file)

# Select one of the context entries for validation

# Define the path to your input and output files
output_file_path = 'spo.jsonl'

# Function to process each entry in your JSON data
def process_entry(original_entry):
    # Extract and transform the 'Prompt' section into 'Context' and 'Instruction'
    print(original_entry.keys())
    context_text = original_entry['Prompt'].split('Context:\n\n')[1].split('Instruction:\n\n')[0]
    instruction_text = original_entry['Prompt'].split('Context:\n\n')[1].split('Instruction:\n\n')[1]
    response_json_str = json.dumps(original_entry['Response'])
    
    # Create the new structure for the entry
    new_entry = {
        'input': context_text,
        'instruction': instruction_text,
        'output': response_json_str  # This assumes 'Response' is a dictionary
    }

    return new_entry

# Main script to read, process, and write the data
with open(input_file_path, 'r', encoding='utf-8') as input_file, \
     open(output_file_path, 'w', encoding='utf-8') as output_file:
    # Load the original JSON data
    original_data = json.load(input_file)
    #print(original_data.keys())

    # Process each entry and write to the new JSONL file
    for entry in original_data:
        #element = original_data[entry]
        processed_entry = process_entry(entry)
        output_file.write(json.dumps(processed_entry) + '\n')  # Write each entry as a single line in the JSONL file