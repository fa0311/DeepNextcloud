import logging
import warnings
from logging import getLogger

import cv2
import numpy as np
from PIL import Image

from src.config import Settings
from src.manager import ModelManager, TagManager
from src.nextcloud import NextCloud


def preprocess_image(image: cv2.typing.MatLike) -> list[cv2.typing.MatLike]:
    scale = 2
    margin = (42 * scale) + 1
    image = image[margin:-margin, margin:-margin]
    res = []
    height, width = image.shape[:2]
    last_y = 0

    for y in range(height):
        if np.mean(image[y]) > 250:
            if y - last_y > width // 2:
                res.append(image[last_y + 1 : y - 1])
            last_y = y
    return res


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    env = Settings()
    logger = getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    logger.info("Starting DeepNextcloud")

    client = NextCloud(env.url, env.username, env.password)

    images = client.recursive_path_list(env.path)
    model_manager = ModelManager(
        url="https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt",
        filename="model-resnet_custom_v3.pt",
        logger=logger,
    )

    tag_manager = TagManager(client)
    invisible_tag_id = tag_manager.get_tag_id(env.invisible_tags, hidden=True)
    rating_list = ["rating:safe", "rating:questionable", "rating:explicit"]

    filename = "1806971333621973454"

    for timestamp, content_type, id, image, displayname, tags in images:
        if content_type not in ["image/png", "image/jpeg", "video/mp4"]:
            continue
        if not image.endswith(f"{filename}.png"):
            continue
        logger.info(f'Starting processing image "{image}"')

        content = client.download(id)
        model = model_manager.load_model()

        img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs = preprocess_image(img)
        tags = set()
        rating = rating_list[0]
        for i, img in enumerate(imgs):
            img_pil = Image.fromarray(img)
            img_pil.save(f"{filename}-{i}.png")
