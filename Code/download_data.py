import os
import synapseclient
import zipfile
import shutil

def download_and_unzip_data():
    """
    Handles the authentication and download of the dataset from Synapse.
    """

    stk = os.environ.get("SYNAPSE_AUTH_TOKEN")

    if stk:
        print("Synapse token loaded successfully from environment variable.")
    else:
        stk = input("Enter your Synapse authentication token: ").strip()
        if not stk:
            raise RuntimeError("No Synapse token provided. Please enter a valid token.")
        print("Synapse token entered successfully.")

    # --- Synapse Login ---
    syn = synapseclient.Synapse()
    try:
        syn.login(authToken=stk, silent=True)
        print("Synapse login successful.")
    except Exception as e:
        print(f"Synapse login failed. Please ensure your authToken is correct. Error: {e}")
        raise

    # --- File & Directory Setup ---
    idc = ["syn51514132"]
    destination_dir = './BRATS/train'
    os.makedirs(destination_dir, exist_ok=True)

    # --- Helper Functions for Download & Unzip ---
    def unzip_data(zip_path, extract_to):
        print(f"Checking for unzipped data at {extract_to}...")
        if os.path.exists(extract_to) and len(os.listdir(extract_to)) > 50:
            print("Data appears to be already unzipped. Skipping.")
            return
        print(f"Unzipping {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Unzipping complete.")

        # Handle nested folders
        content_list = os.listdir(extract_to)
        if len(content_list) == 1 and os.path.isdir(os.path.join(extract_to, content_list[0])):
            nested_folder = os.path.join(extract_to, content_list[0])
            print(f"Moving contents from nested folder '{nested_folder}'...")
            for item in os.listdir(nested_folder):
                shutil.move(os.path.join(nested_folder, item), extract_to)
            os.rmdir(nested_folder)

    # --- Download & Unzip Execution ---
    unzipped_path = os.path.join(destination_dir)
    if os.path.exists(unzipped_path) and len(os.listdir(unzipped_path)) > 500:
        print("Dataset already downloaded and unzipped. Skipping download.")
    else:
        print(f"--- Starting Download: {idc[0]} ---")
        dat = syn.get(entity=idc[0])
        unzip_data(dat.path, destination_dir)
        os.remove(dat.path)
        print(f"Deleted zip file: {dat.path}")
    print("--- Data Preparation Finished ---")

if __name__ == "__main__":
    download_and_unzip_data()