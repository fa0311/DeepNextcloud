import io
import logging
import os
import warnings
import xml.etree.ElementTree as ET
from logging import getLogger

import numpy as np
import requests
import torch
import tqdm
from PIL import Image
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from deep_danbooru_model import DeepDanbooruModel


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


class DeepDanbooru:
    @staticmethod
    def load_model(filename):
        model = DeepDanbooruModel()
        model.load_state_dict(torch.load(filename))
        model.eval()
        model.half()
        model.cuda()
        return model

    @staticmethod
    def preprocess_image(image: Image.Image, size=(512, 512)):
        pic = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        return np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255.0

    @staticmethod
    def predict_and_parse(
        model: DeepDanbooruModel,
        img: np.ndarray,
        score_threshold: float,
    ):
        with torch.no_grad(), torch.autocast("cuda"):
            x = torch.from_numpy(img).cuda()
            y = model(x)[0].detach().cpu().numpy()
            for n in tqdm.tqdm(range(10)):
                model(x)

        rating_dict = {}
        probability_dict = {}
        for tag, probability in zip(model.tags, y):
            if tag.startswith("rating:"):
                rating_dict[tag] = probability
            elif probability >= score_threshold:
                probability_dict[tag] = probability

        rating = max(rating_dict.keys(), key=lambda k: rating_dict[k])
        return list(probability_dict.keys()) + [rating]


class NextCloud:
    def __init__(self, url, username, password):
        self.session = requests.Session()
        self.username = username
        self.password = password
        self.session.auth = (username, password)
        self.url = url

    def request(self, method: str, path: str, tags: list):
        root = ET.Element(
            "d:propfind",
            {
                "xmlns:d": "DAV:",
                "xmlns:oc": "http://owncloud.org/ns",
                "xmlns:nc": "http://nextcloud.org/ns",
            },
        )
        prop = ET.SubElement(root, "d:prop")
        for tag in tags:
            ET.SubElement(prop, tag)

        tag_namespace = {
            "d": "DAV:",
            "oc": "http://owncloud.org/ns",
            "nc": "http://nextcloud.org/ns",
        }
        response = self.session.request(method, path, data=ET.tostring(root))
        root = ET.fromstring(response.text)
        tuples = []
        for response in root.findall(".//d:response", tag_namespace):
            status = response.find(".//d:status", tag_namespace)
            if status is not None and status.text == "HTTP/1.1 200 OK":
                elem = []
                for tag in tags:
                    tag_elem = response.find(f".//{tag}", tag_namespace)
                    if tag_elem is None:
                        raise Exception(f"Tag {tag} not found")
                    elif tag_elem.text is None:
                        tag_elems = response.findall(f".//{tag}/*", tag_namespace)
                        elem.append([tag_elem.text for tag_elem in tag_elems])
                    else:
                        elem.append(tag_elem.text)
                tuples.append(tuple(elem))
        return tuples

    def list(self, path):
        res = self.request(
            "PROPFIND",
            f"{self.url}/remote.php/dav/files/{self.username}/{path}",
            [
                "d:getlastmodified",
                "d:getcontenttype",
                "oc:fileid",
                "d:href",
                "nc:system-tags",
            ],
        )
        return res

    def download(self, id):
        response = self.session.request(
            "GET",
            f"{self.url}/core/preview",
            params={"fileId": id, "a": "true", "x": 3840, "y": 2160},
        )
        return response.content

    def get_tags(self):
        res = self.request(
            "PROPFIND",
            f"{self.url}/remote.php/dav/systemtags/",
            [
                "oc:id",
                "oc:display-name",
            ],
        )
        return res

    def create_tag(
        self,
        name,
        user_visible=True,
        user_assignable=True,
        can_assign=True,
    ):
        self.session.request(
            "POST",
            self.url + "/remote.php/dav/systemtags/",
            json={
                "userVisible": user_visible,
                "userAssignable": user_assignable,
                "canAssign": can_assign,
                "name": name,
            },
        )

    def assign_tag(self, file_id, tag_id):
        response = self.session.request(
            "PUT",
            self.url + f"/remote.php/dav/systemtags-relations/files/{file_id}/{tag_id}",
        )
        return response.text

    def unassign_tag(self, file_id, tag_id):
        response = self.session.request(
            "DELETE",
            self.url + f"/remote.php/dav/systemtags-relations/files/{file_id}/{tag_id}",
        )
        return response.text


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


def remove_all_tags(client: NextCloud):
    images = client.list(env.path)
    tag_manager = TagManager(client)
    for _, _, id, _, tags in images:
        for tag_name in tags:
            tag_id = tag_manager.get_tag_id(tag_name, hidden=False)
            client.unassign_tag(id, tag_id)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    env = Settings()
    logger = getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    logger.info("Starting DeepNextcloud")

    client = NextCloud(env.url, env.username, env.password)
    # remove_all_tags(client)

    images = client.list(env.path)
    model_manager = TagManager.ModelManager(
        url="https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt",
        filename="model-resnet_custom_v3.pt",
        logger=logger,
    )

    tag_manager = TagManager(client)
    invisible_tag_id = tag_manager.get_tag_id(env.invisible_tags, hidden=True)

    for timestamp, content_type, id, image, tags in images:
        if content_type not in ["image/png", "image/jpeg", "video/mp4"]:
            continue
        if env.invisible_tags in tags:
            continue
        logger.info(f'Starting processing image "{image}"')

        content = client.download(id)
        img = Image.open(io.BytesIO(content))
        img = DeepDanbooru.preprocess_image(img)
        model = model_manager.load_model()

        tags = DeepDanbooru.predict_and_parse(
            model,
            img,
            env.score_threshold,
        )

        logger.info(f'Assigning tags "{tags}"')
        for tag_name in tags:
            tag_id = tag_manager.get_tag_id(tag_name, hidden=False)
            client.assign_tag(id, tag_id)
        client.assign_tag(id, invisible_tag_id)
