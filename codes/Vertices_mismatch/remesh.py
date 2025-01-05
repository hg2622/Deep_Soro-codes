import os
import subprocess
import time

os.environ["OMP_NUM_THREADS"] = "8"

def process_files_in_order(input_folder, max_step=100):
    """
    Processes files in order from step_0.obj to step_{max_step}.obj.
    Retries any failed file until it succeeds.
    """
    for step in range(max_step + 1):  # Include step_100
        file_name = f"step_{step}.obj"
        file_path = os.path.join(input_folder, file_name)

        # Ensure the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}. Skipping.")
            continue

        # Process the current file
        while True:  # Retry until successful
            try:
                print(f"Processing: {file_path}")
                process_file_in_subprocess(file_path)
                print(f"Successfully processed: {file_name} ({step}/{max_step})")
                time.sleep(2)               
                break  # Exit retry loop on success
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                print("Retrying in 5 seconds...")
                time.sleep(5)  # Wait before retrying

    print(f"All files processed successfully: {max_step + 1} files.")


def process_file_in_subprocess(input_file):
    """
    Executes the mesh processing function for a single file in a subprocess.
    """
    script = f"""
import pymeshlab

def process_and_combine_mesh(input_file):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_file)
    ms.apply_filter('meshing_remove_unreferenced_vertices')
    ms.apply_filter('generate_splitting_by_connected_components')
    ms.set_current_mesh(1)  # First component
    first_component = ms.current_mesh()
    ms.set_current_mesh(2)  # Second component
    ms.apply_filter(
        'generate_surface_reconstruction_screened_poisson',
        depth=5,
        samplespernode=2,
        pointweight=4,
        preclean=True
    )
    second_component = ms.current_mesh()
    combined_ms = pymeshlab.MeshSet()
    combined_ms.add_mesh(first_component)
    combined_ms.add_mesh(second_component)
    combined_ms.apply_filter('generate_by_merging_visible_meshes')
    combined_ms.save_current_mesh(input_file)  # Overwrite the original file

# Process the file
process_and_combine_mesh(r"{input_file}")
"""
    try:
        result = subprocess.run(
            ["python", "-c", script],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise subprocess.SubprocessError(result.stderr)
    except Exception as e:
        raise RuntimeError(f"Subprocess failed for {input_file}: {e}")


# Execute the function directly
input_folder = '/Users/kevinguo/Desktop/polyfem_project/output/sandtable/sandtable_1_0/original_obj'  # Path to folder containing .obj files
process_files_in_order(input_folder, max_step=100)
