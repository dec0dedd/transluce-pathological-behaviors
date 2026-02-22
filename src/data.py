import msgspec
from typing import Literal

class DatasetItem(msgspec.Struct):
    behavior_id: str
    behavior: str
    category: Literal["drug", "chemical", "biological", "radiological", "nuclear", "explosive"]
    optimizer_target: str

def load_dataset(file_path: str) -> list[DatasetItem]:
    with open(file_path, "rb") as f:
        return [msgspec.json.decode(line, type=DatasetItem) for line in f]