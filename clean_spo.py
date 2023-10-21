
import json
import argparse
import pandas as pd  # Import the pandas library
import re
import demjson
import numpy as np

def extract_from_triplet(triplet_section, field):
    try:
        # Extract the content after the field name, up to the next comma or closing brace.
        content = triplet_section.split(f'"{field}": ')[1]
        value = content.split(",")[0] if ',' in content else content.split("}")[0].replace('[','').replace(']','').replace('   ',' ').replace('  ',' ').replace("\n    \"",'')
        
        # Remove any surrounding quotes from the value.
        value = value.strip(' "')
        return value
    except IndexError:
        # If the field isn't found, return an empty string.
        return ""

def load_and_extract_json(file_path):
    responses = []

    # Load the entire file content
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    # Try to decode the entire content as JSON
    entire_content = json.loads(data)
    for item in entire_content:
        response_json_str_ = item.get('Prompt') + item.get("Response")
        prompt = item.get('Prompt').split('Response:')[0]
        response_json_str = response_json_str_.split('Response:')[1]

        # Initialize a dictionary to hold the various phases and arrays.
        phases_dict = {
            "Entities Phase": [],
            "Entity Opposites Phase": [],
            "Predicates": [],
            "Predicate Opposites Phase": []
        }

        # Process the arrays in the response string
        for phase in phases_dict.keys():
            try:
                # Extract the array part of the string, assuming the array is between '[' and ']'.
                # This approach might be vulnerable to inconsistencies in formatting.
                array_section = response_json_str.partition(f'"{phase}": [')[2].partition('],')[0]

                # Remove any newline characters within the array section, which can interfere with parsing.
                array_section = array_section.replace('\n', '')

                # Now, we need to ensure that we correctly split the items and clean them up.
                # The split(',') method might be too simplistic if there are commas within the items themselves.
                array_items_dirty = array_section.split(',')

                # Clean up the items: strip spaces, remove quotes, and handle any other formatting issues.
                array_items = [item.strip().strip('"') for item in array_items_dirty if item.strip()]

                # Check if the array is not empty or doesn't contain empty strings only.
                if array_items and any(item for item in array_items):
                    phases_dict[phase] = array_items

            except Exception as e:
                phases_dict[phase] = np.nan
            
        # Initialize a dictionary to hold the triplet data for each section.
        triplets_dict = {
                "Triplet 1": {},
                "Triplet 2": {},
                "Triplet 3": {}
        }

        for triplet_number in range(1, 4):
            # Extract the triplet section as text.
            try:
                triplet_section = response_json_str.split(f'"Triplet {triplet_number}": {{')[1].split("}")[0] + "}"  # We are ensuring to get the full triplet data
                    
                # Extract information from the triplet section for each field.
                subject = extract_from_triplet(triplet_section, "Subject")
                predicate = extract_from_triplet(triplet_section, "Predicate")
                object_ = extract_from_triplet(triplet_section, "Object")  # Avoiding keyword 'object'.
                justification = extract_from_triplet(triplet_section, "Justification")
                likelihood = extract_from_triplet(triplet_section, "Likelihood")

                # Create a dictionary that represents the current triplet's information.
                triplet_info = {
                   "Subject": subject,
                   "Predicate": predicate,
                   "Object": object_,
                   "Justification": justification,
                   "Likelihood": likelihood
                }

                # Store this dictionary in the corresponding entry of the 'triplets_dict'.
                triplets_dict[f"Triplet {triplet_number}"] = triplet_info
            except IndexError:  # If the triplet isn't found, move on to the next one.
                triplets_dict[f"Triplet {triplet_number}"] = ''

        # Extract conclusion phase
        try:
            #conclusion = response_json_str.split('\"Conclusion Phase\":')[1].strip()
            conclusion_start = response_json_str.index('\"Conclusion Phase\":') + len('\"Conclusion Phase\":')
            conclusion_end = response_json_str.index("}", conclusion_start)  # finding the end marker for the conclusion phase
            conclusion = response_json_str[conclusion_start:conclusion_end]
            conclusion = conclusion.rstrip('"\n')  # Remove the newline character and the double quote at the end.
            conclusion = conclusion.strip(" \n\"")  # Strips spaces, newline characters, and quotation marks
            conclusion = conclusion.replace("\\n", "\n")
            conclusion = conclusion.replace('   ',' ').replace('  ',' ')
            #conclusion = conclusion_content.strip(" \"\n")
            conclusion = conclusion.replace("\",\n        \"", "\n")  # Handles the case for multiple items.
            conclusion = conclusion.replace("[\n        \"", "")  # Remove the array notation artifacts at the beginning.
            conclusion = conclusion.replace("\"\n    ", "")
            conclusion = conclusion.replace("[",'').replace("]",'')
            conclusion = conclusion.replace("\", \"", " ")
            conclusion = conclusion.replace('", "', ' ')
            pattern = r'"\s*,\s*"'
            conclusion = re.sub(pattern, ' ', conclusion)
            pattern = r'\n\s*"'
            conclusion = re.sub(pattern, '', conclusion)
            pattern = r'\n\s*"'
            conclusion = re.sub(pattern, '', conclusion)
            pattern = r'"\n\s*'
            conclusion = re.sub(pattern, '', conclusion)
            #conclusion = conclusion.replace("\\", "")#.replace('\n')
            #conclusion = conclusion.split("\", \"")#.replace("\", \"", " "
            #conclusion = " ".join(conclusion)
            #conclusion = conclusion.replace(string,' ')
            #conclusion = ' '.join(conclusion.split())
        except:
                conclusion = ''
                # Structure the final response to include the conclusion and the triplets.
                # Combine all the processed information into a single dictionary.
        response_structure = {
        "Prompt": prompt,
        "Response": {
            "Entities Phase": phases_dict["Entities Phase"],
            "Entity Opposites Phase": phases_dict["Entity Opposites Phase"],
            "Predicates": phases_dict["Predicates"],
            "Predicate Opposites Phase": phases_dict["Predicate Opposites Phase"],
            "Premise Phase": triplets_dict,  # This is the dictionary from your previous triplet processing
            "Conclusion Phase": conclusion  # This is from your previous conclusion processing
            }
        }
        responses.append(response_structure)

    return responses

def save_to_json(responses, output_file):
    # Write the list of responses to a JSON file, maintaining their structure.
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a JSON file.')
    parser.add_argument('file_path', type=str, help='Path to the file containing the JSON')
    parser.add_argument('output_file', type=str, help='Path to the output file to save the results')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    file_path = args.file_path
    output_file = args.output_file

    responses = load_and_extract_json(file_path)  # Assuming load_and_extract_json is defined elsewhere
    save_to_json(responses, output_file)
    print(f"Processed and saved responses to {output_file}")

