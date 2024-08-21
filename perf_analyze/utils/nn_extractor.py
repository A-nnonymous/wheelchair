import os
import re
import sys
from typing import List

def parse_csv_for_paths(case_dir,csv_file):
    paths2Tags = {}
    with open(csv_file, 'r') as file:
        for line in file:
            parts = line.strip().split('^')
            #file_path = case_dir.join(parts) + ".py"
            file_path = os.path.join(case_dir, *parts[:])
            model_tag = '-'.join(parts[-2:])
            paths2Tags[file_path ] = model_tag
    return paths2Tags


def extract_and_rename_class(file_path: str, target_class_name: str) -> str:
    class_definition = []  
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        in_class_definition = False
        start_index = None

        for index, line in enumerate(lines):
            line = line.strip()
            if line.startswith('class LayerCase(paddle.nn.Layer):'):
                in_class_definition = True
                start_index = index
                class_definition.append(f'---------------------------------------{target_class_name}---------------------------------------\n')
                continue

            if in_class_definition:
                if any(line.startswith(prefix) for prefix in ['class ', '@']):
                    in_class_definition = False
                else:
                    class_definition.append(lines[index])  

                if not line and not in_class_definition:
                    break  

        if in_class_definition:
            print(f"Warning: Class definition in {file_path} ends abruptly.")

        return ''.join(class_definition)

def main(directory: str, csv_file):
    output_lines = []
    import_lines = []

    paths2Tags = parse_csv_for_paths(case_dir, csv_file)
    for path, tag in paths2Tags.items():
        class_definition = extract_and_rename_class(path + ".py", tag)
        if class_definition:
            output_lines.append(class_definition)

    with open("./nn_with_problem.py", 'w', encoding='utf-8') as output_file:
        for line in output_lines:
            output_file.write(line)


if __name__ == '__main__':
    case_dir = sys.argv[1] if len(sys.argv) > 1 else ''
    csv_file = sys.argv[2] if len(sys.argv) > 2 else ''
    main(case_dir, csv_file)
