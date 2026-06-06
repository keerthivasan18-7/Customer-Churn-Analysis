import os
import subprocess
import shutil


def download_with_kaggle_cli():
    cmd = ["kaggle", "datasets", "download", "-d", "blastchar/telco-customer-churn", "-p", "data", "--unzip"]
    try:
        subprocess.check_call(cmd)
        return True
    except Exception:
        return False


def download_with_kaggle_package():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception:
        return False
    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("blastchar/telco-customer-churn", path="data", unzip=True)
        return True
    except Exception:
        return False


def main():
    os.makedirs("data", exist_ok=True)
    target = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if os.path.exists(target):
        print(f"Dataset already exists at {target}")
        return

    print("Attempting to download with Kaggle CLI...")
    if download_with_kaggle_cli():
        print("Downloaded with Kaggle CLI")
        return

    print("Kaggle CLI failed or not found — trying kaggle package...")
    if download_with_kaggle_package():
        print("Downloaded using kaggle Python package")
        return

    print("Automatic download failed. Please ensure kaggle is installed and credentials are set.")
    print("See README for instructions to set up kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY environment variables.")


if __name__ == "__main__":
    main()
