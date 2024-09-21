import logging
import warnings
from logging import getLogger

import cv2
import numpy as np
from PIL import Image

from src.config import Settings
from src.deep_danbooru import DeepDanbooru
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

    model_manager = ModelManager(
        url="https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt",
        filename="model-resnet_custom_v3.pt",
        logger=logger,
    )

    tag_manager = TagManager(client)
    invisible_tag_id = tag_manager.get_tag_id(env.invisible_tags, hidden=True)
    rating_list = ["rating:safe", "rating:questionable", "rating:explicit"]

    path_list_list = [client.recursive_path_list(x) for x in env.path.split(",")]
    images = [item for sublist in path_list_list for item in sublist]

    for timestamp, content_type, id, image, displayname, tags in images:
        if content_type not in ["image/png", "image/jpeg", "video/mp4"]:
            continue
        if env.invisible_tags in tags:
            continue
        logger.info(f'Starting processing image "{image}"')

        content = client.download(id)
        model = model_manager.load_model()

        img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        imgs = preprocess_image(img)
        tags = set()
        rating = rating_list[0]
        for img in imgs:
            img = DeepDanbooru.preprocess_image(Image.fromarray(img))
            new_tags, new_rating = DeepDanbooru.predict_and_parse(
                model=model,
                img=img,
                score_threshold=env.score_threshold,
            )
            tags.update(new_tags)
            if new_rating and rating_list.index(new_rating) > rating_list.index(rating):
                rating = new_rating

        logger.info(f'Assigning tags "{tags}" and rating "{rating}"')
        tags.add(rating)
        for tag_name in tags:
            tag_id = tag_manager.get_tag_id(tag_name, hidden=False)
            client.assign_tag(id, tag_id)
        client.assign_tag(id, invisible_tag_id)
