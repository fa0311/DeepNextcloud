import logging
import tempfile
import warnings
from logging import getLogger

import cv2
import numpy as np
from PIL import Image

from src.config import Settings
from src.deep_danbooru import DeepDanbooru
from src.manager import ModelManager, TagManager
from src.nextcloud import NextCloud

RgbImage = cv2.typing.MatLike


def split_screenshot_panels(image: RgbImage) -> list[RgbImage]:
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


def decode_image_rgb(content: bytes) -> RgbImage:
    bgr = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise Exception("could not decode image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def sample_video_frames_rgb(content: bytes, positions: list[float]) -> list[RgbImage]:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(content)
        f.flush()
        capture = cv2.VideoCapture(f.name)
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise Exception("could not read video frame count")
            frames = []
            for position in positions:
                capture.set(cv2.CAP_PROP_POS_FRAMES, min(int(frame_count * position), frame_count - 1))
                ok, bgr = capture.read()
                if ok and bgr is not None:
                    frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
    if not frames:
        raise Exception("could not decode any video frame")
    return frames


def images_to_tag(content: bytes, content_type: str, video_frame_positions: list[float]) -> list[RgbImage]:
    if content_type == "video/mp4":
        return sample_video_frames_rgb(content, video_frame_positions)
    image = decode_image_rgb(content)
    return split_screenshot_panels(image) or [image]


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    env = Settings()
    logger = getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    logger.info("Starting DeepNextcloud")

    client = NextCloud(env.url, env.username, env.password, concurrency=env.tag_assign_concurrency)

    model_manager = ModelManager(
        url="https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt",
        filename="model-resnet_custom_v3.pt",
        logger=logger,
    )
    model = model_manager.load_model()

    tag_manager = TagManager(client)
    invisible_tag_id = tag_manager.get_tag_id(env.invisible_tags, hidden=True)
    rating_list = ["rating:safe", "rating:questionable", "rating:explicit"]

    path_list_list = [client.recursive_path_list(x) for x in env.path.split(",")]
    images = [item for sublist in path_list_list for item in sublist]

    for timestamp, content_type, id, image, displayname, tags in images:
        try:
            if content_type not in ["image/png", "image/jpeg", "video/mp4"]:
                continue
            if env.invisible_tags in tags:
                continue
            logger.info(f'Starting processing image "{image}"')

            content = client.download(image)

            imgs = images_to_tag(content, content_type, env.video_frame_positions)
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
            tag_ids = [tag_manager.get_tag_id(tag_name, hidden=False) for tag_name in tags]
            client.assign_tags(id, tag_ids)
            client.assign_tag(id, invisible_tag_id)
        except Exception as e:
            logger.error(f'Error processing image "{image}": {e}')
