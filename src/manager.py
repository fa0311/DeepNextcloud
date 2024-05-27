import os
from logging import getLogger

import torch

from src.deep_danbooru import DeepDanbooru
from src.nextcloud import NextCloud
from TorchDeepDanbooru.deep_danbooru_model import DeepDanbooruModel


class TagManager:
    def __init__(self, tag: NextCloud):
        self.tag = tag
        self.tags = self.tag.get_tags()

    def get_tag_id(self, name: str, hidden=False):
        for tag_id, tag_name in self.tags:
            if tag_name == name:
                return tag_id
        self.tag.create_tag(
            name,
            user_visible=not hidden,
            user_assignable=not hidden,
            can_assign=True,
        )
        self.tags = self.tag.get_tags()
        for tag_id, tag_name in self.tags:
            if tag_name == name:
                return tag_id
        raise Exception("Tag not found")


class ModelManager:
    def __init__(self, filename: str, url: str, logger=None):
        self.filename = filename
        self.url = url
        self.model = None
        self.tagger_tags = None
        self.logger = logger or getLogger(__name__)

    def load_model(self) -> DeepDanbooruModel:
        if self.model is None:
            self.logger.info("Loading model")
            if not os.path.exists(self.filename):
                torch.hub.download_url_to_file(self.url, self.filename)
            self.model = DeepDanbooru.load_model(
                filename=self.filename,
            )
            self.logger.info("Model loaded")
        return self.model
