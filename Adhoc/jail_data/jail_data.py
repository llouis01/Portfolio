import os
import json
import base64
import requests
import numpy as np
import pandas as pd
import random as rd
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor as TPE, as_completed

############ START Functions ############
# docker run -it -v C:/Users/RoiMinuit/Desktop/data/mugshots:/mugshots jail-data-scrapper #

# set base url and create folder to hold images
mug_base = "https://www.horrycountysc.gov/apps/bookings/mugshots"
mug_path = "/mugshots"


# func to grab all data between two dates
def get_jail_data(start_date, end_date, county):
    data = []
    while start_date <= end_date:
        payload = {
            "bookingDate": start_date.strftime("%Y-%m-%dT04:00:00.000Z"),
            "county": county
        }
        headers= {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        base_url = "https://www.horrycountysc.gov/apps/bookings/api/bookings/search"
        response = requests.post(base_url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            for row in response.json():
                data.append(row)
        else:
            print(response.status_code)
        start_date += timedelta(days=1)
    return data



# define worker function
def download_mugshot(inmate, mug_base):
    idx = inmate.get("id")
    if not idx:
        return f"Skipped; missing id"
    
    # check if image already downloaded
    path = f"{mug_path}/{idx}.jpg"
    if os.path.exists(path):
        return f"Already downloaded ->>"
    
    # downloaded mugshot
    mug_url = f"{mug_base}/{idx}"
    response = requests.get(mug_url)

    # ensure 200 response, then write file to mugshot folder
    if response.status_code == 200:
        with open(f"/mugshots/{idx}.jpg", "wb") as f:
            f.write(response.content)
            print(f"Downloaded to {path}")
    else:
        return f"Failed {idx}"
    
############ END Functions ############


# variables
start_date = datetime(2024, 4, 13, tzinfo=timezone.utc)
end_date = datetime.now(timezone.utc)
county = 'horry'


# get booking data from Horry PD
data = get_jail_data(start_date, end_date, county)


# run with threaded execution
with TPE(max_workers=25) as executor:
    futures = [executor.submit(download_mugshot, inmate, mug_base) for inmate in data]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading mugshots..."):
        print(future.result())


# make df for future work
df = pd.DataFrame(data).reset_index()
df.drop(columns='index', inplace=True)


# grab booking id
idx = [id for id in df.id]
b64 = {}

# add encoding to separate df
for i in range(len(idx)):
    try:
        with open(f"/mugshots/{str(idx[i])}.jpg", "rb") as img_file:
            b64[idx[i]] = base64.b64encode(img_file.read())
    except Exception as e:
        print(e)


# df b64 data
b64_df = pd.DataFrame({"b64_code": b64})
b64_df.reset_index(inplace=True)
b64_df.rename(columns={'index': 'id'}, inplace=True)


# store encoded images in df
final_df = df.merge(b64_df, on='id', how='left')
final_df.to_csv(f"/mugshots/jail_data.csv", index=False)

print("Current working dir:", os.getcwd())
print("Mugshot path:", os.path.abspath("mugshots"))