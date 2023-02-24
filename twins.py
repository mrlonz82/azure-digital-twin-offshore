import os
import time
import uuid
import datetime
import pandas as pd
from azure.digitaltwins.core import DigitalTwinsClient
from azure.identity import InteractiveBrowserCredential, TokenCachePersistenceOptions
import dotenv


def authorize():
    credential = InteractiveBrowserCredential(cache_persistence_options=TokenCachePersistenceOptions())
    client = DigitalTwinsClient(os.getenv("AZURE_URL"), credential)
    return client


def create_twin(date_time, path_to_file: str, client):
    # Define the digital twin to upload
    model_id = os.getenv("DIGITAL_TWINS_ID")
    digital_twin_id = 'WindTurbine-' + str(uuid.uuid4())
    data: pd.DataFrame = _read_cev_for_data(date_time, path_to_file)
    if not data.empty:
        _windSpeed = float(data["wind speed"].values)
        _windDirection = float(data["wind direction"].values)
        _datetime = data["date"].values[0]
        temporary_twin = {
            "$metadata": {
                "$model": model_id
            },
            "$dtId": digital_twin_id,
            "windSpeed": _windSpeed,
            "windDirection": _windDirection,
            "datetime": datetime.datetime.strptime(_datetime, "%d/%m/%Y %H:%M")
        }

        created_twin = client.upsert_digital_twin(digital_twin_id, temporary_twin)
        print('Created Digital Twin:')
        print(created_twin)
        print(f"Digital twin {digital_twin_id} uploaded successfully.")


def _read_cev_for_data(date_time, path_to_file: str):
    df = pd.read_csv(path_to_file, dtype={"date": str, "wind speed": str, "wind direction": str})
    df = df.loc[df["date"] == date_time.strftime(format="%d/%m/%Y %H:%M")]
    return df


def main():
    # load environment variables
    dotenv.load_dotenv(".env")

    # set the time of the data
    date = datetime.datetime(2007, 1, 1, 0, 0)
    client = authorize()
    while True:
        create_twin(date, path_to_file="datasets/offshore wind farm data.csv", client=client)
        date += datetime.timedelta(minutes=10)

        # sleep for 10 minutes then upload the next
        time.sleep(600)


main()
