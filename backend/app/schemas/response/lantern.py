from typing import List

from pydantic import BaseModel


class LanternResponseModel(BaseModel):
    lantern_id: str
    owner_name: str
    is_current_lantern: bool


class LanternDetailResponseModel(BaseModel):
    lantern_id: str
    owner_name: str
    images: List[str]
    background_sounds: List[str]
