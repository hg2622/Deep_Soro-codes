import os

# Function to read the face information from a text file
def read_faces_from_txt(filepath):
    faces = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith("f "):
                faces.append(line.strip())
    return faces

# Function to process a single OBJ file
def process_obj_file(obj_file_path, face_info_path):
    with open(obj_file_path, 'r') as file:
        lines = file.readlines()

    # Find the index of the first face line
    first_face_index = next((i for i, line in enumerate(lines) if line.startswith("f ")), None)
    if first_face_index is None:
        print(f"No face information found in {obj_file_path}. Skipping file.")
        return

    # Calculate the new index
    new_index = first_face_index + 2108

    # Remove all face information from the new index onward
    modified_lines = lines[:new_index]

    # Ensure all remaining lines end with a newline
    modified_lines = [line if line.endswith("\n") else line + "\n" for line in modified_lines]

    # Read the new face information from the provided file
    new_faces = read_faces_from_txt(face_info_path)

    # Append the new face information
    for face in new_faces:
        modified_lines.append(face + "\n")

    # Save the updated OBJ file
    with open(obj_file_path, 'w') as file:
        file.writelines(modified_lines)
    print(f"Updated {obj_file_path} successfully.")

# Function to find and process all folders named "sandtable_x_x" and their OBJ files
def process_sandtable_folders(base_path, face_info_path):
    for folder_name in os.listdir(base_path):
        if folder_name.startswith("sandtable_"):  # Check for folders with the name pattern
            sandtable_path = os.path.join(base_path, folder_name)
            original_obj_path = os.path.join(sandtable_path, "original_obj")  # Navigate to "original_obj"
            if os.path.isdir(original_obj_path):  # Ensure it's a folder
                print(f"Processing folder: {original_obj_path}")
                # Process all OBJ files inside this folder
                process_all_obj_files(original_obj_path, face_info_path)

# Function to process all OBJ files in a folder
def process_all_obj_files(folder_path, face_info_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".obj"):
            obj_file_path = os.path.join(folder_path, filename)
            process_obj_file(obj_file_path, face_info_path)

# Usage Example
base_path = "/Users/kevinguo/Desktop/polyfem_project/output/sandtable"  # Path containing multiple sandtable_x_x folders
face_info_path = "/Users/kevinguo/Desktop/polyfem_project/correct_sequence.txt"  # Path to the text file with new face information
process_sandtable_folders(base_path, face_info_path)
