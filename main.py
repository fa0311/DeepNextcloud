import io
import logging
import warnings
from logging import getLogger

from PIL import Image

from src.config import Settings
from src.deep_danbooru import DeepDanbooru
from src.manager import ModelManager, TagManager
from src.nextcloud import NextCloud

if __name__ == "__main__":
    warnings.simplefilter("ignore")
    env = Settings()
    logger = getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    logger.info("Starting DeepNextcloud")

    client = NextCloud(env.url, env.username, env.password)

    path_list_list = [client.recursive_path_list(x) for x in env.path.split(",")]
    images = [item for sublist in path_list_list for item in sublist]
    model_manager = ModelManager(
        url="https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt",
        filename="model-resnet_custom_v3.pt",
        logger=logger,
    )

    tag_manager = TagManager(client)
    invisible_tag_id = tag_manager.get_tag_id(env.invisible_tags, hidden=True)

    for timestamp, content_type, id, image, displayname, tags in images:
        if content_type not in ["image/png", "image/jpeg", "video/mp4"]:
            continue
        if env.invisible_tags in tags:
            continue
        logger.info(f'Starting processing image "{image}"')

        content = client.download(id)
        img = Image.open(io.BytesIO(content))
        img = DeepDanbooru.preprocess_image(img)
        model = model_manager.load_model()

        tags, rating = DeepDanbooru.predict_and_parse(
            model=model,
            img=img,
            score_threshold=env.score_threshold,
        )

        logger.info(f'Assigning tags "{tags}" and rating "{rating}"')
        tags.append(rating or "rating:safe")
        for tag_name in tags:
            tag_id = tag_manager.get_tag_id(tag_name, hidden=False)
            client.assign_tag(id, tag_id)
        client.assign_tag(id, invisible_tag_id)
