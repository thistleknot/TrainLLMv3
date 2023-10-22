import json
import argparse

# Function to parse and print the content of each JSON line in your file.
def parse_and_print_jsonl(file_path):
    with open(file_path, 'r') as file:
        # Counter for line numbers (optional, for your reference)
        line_number = 1

        for line in file:
            try:
                # Parse the JSON data from the line
                json_obj = json.loads(line)

                # Extract the specific keys
                instruction = json_obj.get('instruction', 'Not found')
                input_data = json_obj.get('input', 'Not found')  # 'input' is a built-in function, so we're using input_data
                output = json_obj.get('output', 'Not found')

                # Print the extracted data
                print(f'Line {line_number}:')
                print('Instruction:', instruction)
                print('Input:', input_data)
                print('Output:', output)
                print('\n')  # Print a newline for better separation between JSONL entries

                line_number += 1  # Move to the next line number

            except json.JSONDecodeError as e:
                # Handle lines that aren't valid JSON
                print(f"Couldn't parse line {line_number}: {e}")
                print('\n')  # Print a newline for better separation between JSONL entries

# Setting up the argument parser
parser = argparse.ArgumentParser(description='Process a .jsonl or .txt file containing JSON lines.')
parser.add_argument('file_path', type=str, help='Path to the input file.')

# Parsing the arguments
args = parser.parse_args()

# Execute the function with your file path
parse_and_print_jsonl(args.file_path)
