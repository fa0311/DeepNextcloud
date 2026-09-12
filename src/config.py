from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    def __init__(self):
        super().__init__()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DEEPNEXTCLOUD_",
    )
    url: str = Field()
    username: str = Field()
    password: str = Field()
    path: str = Field()
    score_threshold: float = Field(default=0.5)
    invisible_tags: str = Field(default="Tagged by deepdanbooru v1.0.0")
    video_frame_positions: list[float] = Field(default=[0.1, 0.5, 0.9])
    tag_assign_concurrency: int = Field(default=8)
