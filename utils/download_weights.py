# Copyright 2025 Taihong Yang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to download pretrained weights for KDLAE and ASDQE models.
"""

import os
import sys
import urllib.request
from urllib.error import URLError, HTTPError


# Weight download URLs and destination paths
WEIGHTS_CONFIG = [
    {
        "name": "KDLAE-T",
        "url": "https://github.com/yangtaihong59/Rethink_Acoustic_Image_Enhancement/releases/download/weight/KDLAE_T.pth",
        "dest": "KDLAE/weights/KDLAE_T.pth"
    },
    {
        "name": "KDLAE-S-FLS",
        "url": "https://github.com/yangtaihong59/Rethink_Acoustic_Image_Enhancement/releases/download/weight/KDLAE-S-FLS.pth",
        "dest": "KDLAE/weights/KDLAE-S-FLS.pth"
    },
    {
        "name": "KDLAE-S-US",
        "url": "https://github.com/yangtaihong59/Rethink_Acoustic_Image_Enhancement/releases/download/weight/KDLAE-S-US.pth",
        "dest": "KDLAE/weights/KDLAE-S-US.pth"
    },
    {
        "name": "ASDQE",
        "url": "https://github.com/yangtaihong59/Rethink_Acoustic_Image_Enhancement/releases/download/weight/ASDQE.pth",
        "dest": "ASDQE/weights/ASDQE.pth"
    }
]


class ProgressBar:
    """Simple progress bar for download progress."""
    
    def __init__(self, total_size):
        self.total_size = total_size
        self.downloaded = 0
        
    def update(self, chunk_size):
        self.downloaded += chunk_size
        percent = (self.downloaded / self.total_size) * 100 if self.total_size > 0 else 0
        bar_length = 40
        filled = int(bar_length * self.downloaded // self.total_size) if self.total_size > 0 else 0
        bar = '=' * filled + '-' * (bar_length - filled)
        size_mb = self.downloaded / (1024 * 1024)
        total_mb = self.total_size / (1024 * 1024) if self.total_size > 0 else 0
        print(f'\r[{bar}] {percent:.1f}% ({size_mb:.2f}/{total_mb:.2f} MB)', end='', flush=True)


def download_file(url, dest_path, weight_name):
    """Download a file with progress bar."""
    dest_dir = os.path.dirname(dest_path)
    
    # Create directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(dest_path):
        file_size = os.path.getsize(dest_path)
        response = None
        try:
            # Check file size on server
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req) as response:
                remote_size = int(response.headers.get('Content-Length', 0))
            
            if file_size == remote_size and remote_size > 0:
                print(f"✓ {weight_name} already exists and is up-to-date ({file_size / (1024*1024):.2f} MB)")
                return True
            else:
                print(f"⚠ {weight_name} exists but size differs. Re-downloading...")
        except Exception as e:
            print(f"⚠ Could not verify existing file. Re-downloading... ({e})")
    
    try:
        print(f"Downloading {weight_name}...")
        
        # Open URL and get file size
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            
            # Download with progress
            progress = ProgressBar(total_size)
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(8192)  # 8KB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    progress.update(len(chunk))
            
            print()  # New line after progress bar
            print(f"✓ Successfully downloaded {weight_name} to {dest_path}")
            return True
            
    except HTTPError as e:
        print(f"\n✗ HTTP Error {e.code}: {e.reason} for {weight_name}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False
    except URLError as e:
        print(f"\n✗ URL Error: {e.reason} for {weight_name}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False
    except Exception as e:
        print(f"\n✗ Error downloading {weight_name}: {str(e)}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def main():
    """Main function to download all weights."""
    print("=" * 60)
    print("Downloading pretrained weights for Rethink Acoustic Image Enhancement")
    print("=" * 60)
    print()
    
    success_count = 0
    total_count = len(WEIGHTS_CONFIG)
    
    for config in WEIGHTS_CONFIG:
        success = download_file(config["url"], config["dest"], config["name"])
        if success:
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"Download complete: {success_count}/{total_count} weights downloaded successfully")
    print("=" * 60)
    
    if success_count < total_count:
        print("\n⚠ Some downloads failed. Please check your internet connection and try again.")
        sys.exit(1)
    else:
        print("\n✓ All weights downloaded successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
