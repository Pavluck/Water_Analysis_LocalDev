"""
NP ˚❀༉‧ 
Retrieve and organize image dataset.
Does this currently with one DOI, from Figshare:
https://figshare.com/articles/dataset/A_Comprehensive_Surface_Water_Quality_Monitoring_Dataset_1940-2023_2_82Million_Record_Resource_for_Empirical_and_ML-Based_Research/27800394/2
Could be optimized to download using API key
"""
# ~~~~ Imports ~~~~
import requests
import os
from tqdm import tqdm # cute task bar 

# ~~~~ Global Functions ~~~~
def figshare_dataset(article_id, output_dir="cnn_image_data"):
  """TODO: Given an article id and name to place files, extracts the files using the API key"""
    # API endpoint
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"
    
    # prevent oopsies: match sure we have the metadata to get file links
    print(f"Fetching metadata for Article ID: {article_id}...")
    response = requests.get(api_url)
    # error handling
    if response.status_code != 200:
        print(f"Error: Could not access Figshare. Status: {response.status_code}")
        return

    data = response.json()
    files = data.get('files', [])
    # go ahead and create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Found {len(files)} files. Starting download...")

    # download each file in the files list with API using a for loop
    for file_info in tqdm(files):
        file_name = file_info['name']
        download_url = file_info['download_url']
        file_path = os.path.join(output_dir, file_name)

        # Download with streaming to handle large files/many images
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

# The ID from DOI: 10.6084/m9.figshare.27800394  <-- that last bit
article_id = "27800394"
# throw it in
download_figshare_dataset(article_id)
