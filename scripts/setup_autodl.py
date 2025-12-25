import os
import zipfile
import sys
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, installing...")
    os.system("pip install tqdm")
    from tqdm import tqdm

def setup_autodl_data():
    """
    Extracts the MABe dataset from autodl-tmp to the project data directory.
    """
    # Determine project root (parent of scripts/)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    
    # Target directory
    data_dir = project_root / 'data'
    
    # Possible locations for the zip file
    zip_candidates = [
        Path('/root/autodl-tmp/MABe-mouse-behavior-detection.zip'), # Standard AutoDL absolute path
        project_root / 'autodl-tmp/MABe-mouse-behavior-detection.zip', # Local relative to project
        Path('autodl-tmp/MABe-mouse-behavior-detection.zip') # Relative to CWD
    ]
    
    zip_path = None
    for candidate in zip_candidates:
        if candidate.exists():
            zip_path = candidate
            break
            
    if zip_path is None:
        print("Error: Could not find 'MABe-mouse-behavior-detection.zip' in standard locations.")
        print(f"Checked: {[str(p) for p in zip_candidates]}")
        print("Please ensure the zip file is uploaded to /root/autodl-tmp/ or the project root.")
        return

    print(f"Found dataset archive: {zip_path}")
    print(f"Target directory: {data_dir}")
    
    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Extracting... This may take a while.")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files to extract
            members = zip_ref.infolist()
            
            # Use tqdm for progress bar
            for member in tqdm(members, desc="Extracting", unit="file"):
                zip_ref.extract(member, data_dir)
                
        print("\nExtraction completed successfully!")
        print(f"Data is ready in {data_dir}")
        
        # Verify extraction
        expected_files = ['train.csv', 'sample_submission.csv']
        found_files = [f.name for f in data_dir.iterdir()]
        print(f"Files in data dir: {found_files}")
        
    except Exception as e:
        print(f"\nError during extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_autodl_data()
