import logging
import warnings
import xml.etree.ElementTree as ET
from logging import getLogger

import cv2
import numpy as np
import onnxruntime as rt
import requests
from huggingface_hub import hf_hub_download
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
    score_threshold: float = Field(default=0.8)
    invisible_tags: str = Field(default="Tagged by deepdanbooru v1.0.0")


class DeepDanbooru:
    @staticmethod
    def load_model(repo_id, filename):
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model = rt.InferenceSession(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        model_meta = model.get_modelmeta().custom_metadata_map
        tags = eval(model_meta["tags"])
        return model, tags

    @staticmethod
    def preprocess_image(image, size=512):
        h, w = image.shape[:-1]
        h, w = (size, int(size * w / h)) if h > w else (int(size * h / w), size)
        ph, pw = size - h, size - w
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        image = cv2.copyMakeBorder(
            image, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2, cv2.BORDER_REPLICATE
        )
        image = image.astype(np.float32) / 255
        return image[np.newaxis, :]

    @staticmethod
    def predict_and_parse(model, image, tags, score_threshold):
        probs = model.run(None, {"input_1": image})[0][0]
        probs = probs.astype(np.float32)
        return [
            label
            for prob, label in zip(probs.tolist(), tags)
            if prob >= score_threshold
        ]


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
                    if tag_elem.text is None:
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
        self, name, user_visible=True, user_assignable=True, can_assign=True
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
        def __init__(self, repo_id: str, filename: str, logger=None):
            self.repo_id = repo_id
            self.filename = filename
            self.tagger_model = None
            self.tagger_tags = None
            self.logger = logger or getLogger(__name__)

        def load_model(self):
            if self.tagger_model is None:
                self.logger.info("Loading model")
                self.tagger_model, self.tagger_tags = DeepDanbooru.load_model(
                    repo_id=self.repo_id,
                    filename=self.filename,
                )
                self.logger.info("Model loaded")
            return self.tagger_model, self.tagger_tags


def remove_all_tags():
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
    images = client.list(env.path)
    model_manager = TagManager.ModelManager(
        repo_id="skytnt/deepdanbooru_onnx",
        filename="deepdanbooru.onnx",
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
        img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = DeepDanbooru.preprocess_image(img)
        tagger_model, tagger_tags = model_manager.load_model()

        tags = DeepDanbooru.predict_and_parse(
            tagger_model,
            img,
            tagger_tags,
            env.score_threshold,
        )

        logger.info(f'Assigning tags "{tags}"')
        for tag_name in tags:
            tag_id = tag_manager.get_tag_id(tag_name, hidden=False)
            client.assign_tag(id, tag_id)
        client.assign_tag(id, invisible_tag_id)
