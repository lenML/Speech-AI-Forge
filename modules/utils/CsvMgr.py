import json
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


class DataExistsError(Exception):
    pass


class DataNotFoundError(Exception):
    pass


# FIXME: 😓这个东西写的比较拉跨，最好找个什么csv库替代掉...
class BaseManager:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.columns = ["id", "name", "desc", "params"]
        if not os.path.exists(csv_file):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.csv_file, index=False)

    def _load_data(self):
        return pd.read_csv(self.csv_file)

    def _save_data(self, df):
        df.to_csv(self.csv_file, index=False)

    def add_item(self, item_id, name, desc, params):
        df = self._load_data()
        if item_id in df["id"].values:
            raise DataExistsError(f"Item ID {item_id} already exists.")
        new_row = pd.DataFrame(
            [
                {
                    "id": item_id,
                    "name": name,
                    "desc": desc,
                    "params": json.dumps(params, ensure_ascii=False),
                }
            ]
        )
        df = pd.concat([df, new_row], ignore_index=True)
        self._save_data(df)

    def delete_item(self, item_id):
        df = self._load_data()
        if item_id not in df["id"].values:
            raise DataNotFoundError(f"Item ID {item_id} not found.")
        df = df[df["id"] != item_id]
        self._save_data(df)

    def update_item(self, item_id, name=None, desc=None, params=None):
        df = self._load_data()
        if item_id not in df["id"].values:
            raise DataNotFoundError(f"Item ID {item_id} not found.")
        if name:
            df.loc[df["id"] == item_id, "name"] = name
        if desc:
            df.loc[df["id"] == item_id, "desc"] = desc
        if params:
            df.loc[df["id"] == item_id, "params"] = params
        self._save_data(df)

    def get_item(self, item_id):
        df = self._load_data()
        if item_id not in df["id"].values:
            raise DataNotFoundError(f"Item ID {item_id} not found.")
        item = df[df["id"] == item_id].to_dict("records")[0]
        item["params"] = json.loads(item["params"])
        return item

    def list_items(self):
        df = self._load_data()
        items = df.to_dict("records")
        for item in items:
            item["params"] = json.loads(item["params"])
        return items

    def find_item_by_name(self, name):
        df = self._load_data()
        if name not in df["name"].values:
            raise DataNotFoundError(f"Name {name} not found.")
        item = df[df["name"] == name].to_dict("records")[0]
        item["params"] = json.loads(item["params"])
        return item

    def find_params_by_name(self, name):
        try:
            return self.find_item_by_name(name)["params"]
        except Exception as e:
            logger.error(e)
            return {}

    def find_params_by_id(self, id):
        try:
            return self.get_item(id)["params"]
        except Exception as e:
            logger.error(e)
            return {}


# Usage example
if __name__ == "__main__":

    class SpeakerManager(BaseManager):
        def __init__(self, csv_file):
            super().__init__(csv_file)

    manager = SpeakerManager("speakers.test.csv")

    try:
        # Add speaker
        manager.add_item(
            1, "Speaker1", "Description for speaker 1", '{"param1": "value1"}'
        )
    except DataExistsError as e:
        print(e)

    # List all speakers
    speakers = manager.list_items()
    print(speakers)

    try:
        # Get specific speaker
        speaker = manager.get_item(1)
        print(speaker)
    except DataNotFoundError as e:
        print(e)

    try:
        # Update speaker
        manager.update_item(
            1, name="Updated Speaker1", desc="Updated description for speaker 1"
        )
    except DataNotFoundError as e:
        print(e)

    try:
        # Delete speaker
        manager.delete_item(1)
    except DataNotFoundError as e:
        print(e)

    try:
        # Find speaker by name
        speaker_by_name = manager.find_item_by_name("Updated Speaker1")
        print(speaker_by_name)
    except DataNotFoundError as e:
        print(e)

    # Find speakers by params
    speakers_by_params = manager.find_items_by_params('{"param1": "value1"}')
    print(speakers_by_params)
