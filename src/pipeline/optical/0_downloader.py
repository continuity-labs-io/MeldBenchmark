import os
import re
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaIoBaseDownload
import io

import logging
logger = logging.getLogger(__name__)

# Local Prototyping Parameters
TARGET_FOLDER_ID = "1IxA6w2Z76wgiy_NNNY72GdwUK-a8VcKQ"  #  shared folder id  https://drive.google.com/drive/folders/[id]
TARGET_CHANNEL = "ch1"
MAX_FRAMES = 10  # Restricted for local 50GB storage limit
DOWNLOAD_DIR = "./dataset/raw_tiffs"


def setup_drive_api():
    """Authenticates and returns the Google Drive API service."""
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    # Ensure credentials.json is loaded from the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    creds_path = os.path.join(script_dir, "credentials.json")
    flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
    creds = flow.run_local_server(port=0)
    return build("drive", "v3", credentials=creds)


def download_mini_crop(service):
    """Downloads a 10-frame sequential crop for local PyTorch prototyping."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    query = f"'{TARGET_FOLDER_ID}' in parents and mimeType='image/tiff'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    pattern = re.compile(rf".*_{TARGET_CHANNEL}_.*_stack(\d{{4}})_.*decon\.tif")

    download_count = 0
    for file in sorted(files, key=lambda x: x["name"]):
        match = pattern.match(file["name"])
        if match:
            stack_num = int(match.group(1))
            if stack_num < MAX_FRAMES:
                logger.info(f"Downloading {file['name']}...")
                request = service.files().get_media(fileId=file["id"])
                file_path = os.path.join(DOWNLOAD_DIR, file["name"])

                with io.FileIO(file_path, "wb") as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while done is False:
                        status, done = downloader.next_chunk()

                download_count += 1
                if download_count >= MAX_FRAMES:
                    break

    logger.info(f"Successfully downloaded {download_count} frames for local testing.")


if __name__ == "__main__":
    drive_service = setup_drive_api()
    download_mini_crop(drive_service)
