import shutil
from pathlib import Path

def consolidate_trajectories(source_dir: str, dest_dir: str):
    """
    Crawls a source directory for CSVs and copies them to a single destination folder.
    """
    # Setup paths
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Create the destination folder if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Recursively find all .csv files in the source directory and its subfolders
    csv_files = list(source_path.rglob("*.csv"))
    
    print(f"[*] Found {len(csv_files)} CSV files. Starting extraction...")
    
    success_count = 0
    for file_path in csv_files:
        # Prefix the filename with its parent folder's name to prevent overwriting
        # e.g., "Run_01" + "_" + "trajectory.csv" -> "Run_01_trajectory.csv"
        safe_filename = f"{file_path.parent.name}_{file_path.name}"
        destination_file = dest_path / safe_filename
        
        try:
            # shutil.copy2 preserves file metadata (timestamps, etc.)
            shutil.copy2(file_path, destination_file)
            success_count += 1
        except Exception as e:
            print(f"[!] Failed to copy {file_path.name}: {e}")

    print(f"[✓] Successfully consolidated {success_count} files into '{dest_dir}'.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Replace these strings with your actual directory paths
    TOP_LEVEL_FOLDER = "data/results/2026-04-07_12-13-41" 
    CONSOLIDATED_FOLDER = "data/results/2026-04-07_12-13-41/consolidated_trajectories"
    # ---------------------
    
    consolidate_trajectories(TOP_LEVEL_FOLDER, CONSOLIDATED_FOLDER)