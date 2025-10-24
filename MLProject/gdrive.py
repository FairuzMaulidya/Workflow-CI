import os
import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# === 1. Load credential service account ===
creds_json = os.environ.get("GDRIVE_CREDENTIALS")
if not creds_json:
    raise ValueError("❌ Environment variable 'GDRIVE_CREDENTIALS' tidak ditemukan.")

creds_info = json.loads(creds_json)
credentials = Credentials.from_service_account_info(
    creds_info,
    scopes=["https://www.googleapis.com/auth/drive"]
)

# === 2. Build Drive API ===
service = build("drive", "v3", credentials=credentials)

# === 3. Shared Drive ID / Folder tujuan ===
SHARED_DRIVE_ID = os.environ.get("GDRIVE_FOLDER_ID")
if not SHARED_DRIVE_ID:
    raise ValueError("❌ Environment variable 'GDRIVE_FOLDER_ID' tidak ditemukan.")

# === 4. Fungsi untuk upload folder dan file ===
def upload_directory(local_dir_path, parent_drive_id):
    """
    Rekursif:
    - Jika folder → buat folder di Drive, lalu upload isinya lagi.
    - Jika file → langsung upload ke folder parent.
    """
    for item_name in os.listdir(local_dir_path):
        item_path = os.path.join(local_dir_path, item_name)
        if os.path.isdir(item_path):
            folder_meta = {
                "name": item_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [parent_drive_id]
            }
            created_folder = service.files().create(
                body=folder_meta,
                fields="id",
                supportsAllDrives=True
            ).execute()
            new_folder_id = created_folder["id"]
            print(f"📁 Created folder: {item_name} (ID: {new_folder_id})")
            upload_directory(item_path, new_folder_id)
        else:
            print(f"📄 Uploading file: {item_name}")
            file_meta = {
                "name": item_name,
                "parents": [parent_drive_id]
            }
            media = MediaFileUpload(item_path, resumable=True)
            service.files().create(
                body=file_meta,
                media_body=media,
                fields="id",
                supportsAllDrives=True
            ).execute()

# === 5. Upload semua run_id di ./mlruns/0 ===
local_mlruns_0 = "./mlruns/0"
if not os.path.exists(local_mlruns_0):
    print("⚠️ Folder mlruns/0 tidak ditemukan, tidak ada yang diupload.")
else:
    for run_id in os.listdir(local_mlruns_0):
        run_path = os.path.join(local_mlruns_0, run_id)
        if os.path.isdir(run_path):
            # Buat folder run_id di root Shared Drive
            folder_meta = {
                "name": run_id,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [SHARED_DRIVE_ID]
            }
            run_folder = service.files().create(
                body=folder_meta,
                fields="id",
                supportsAllDrives=True
            ).execute()
            run_folder_id = run_folder["id"]
            print(f"=== Created run_id folder: {run_id} (ID: {run_folder_id}) ===")
            upload_directory(run_path, run_folder_id)

    print("✅ Semua run_id dan artifact telah berhasil diupload ke Google Drive!")
