import logging
import warnings
from logging import getLogger

import tqdm

from src.config import Settings
from src.manager import TagManager
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
    tag_manager = TagManager(client)
    for _, _, id, _, _, tags in tqdm.tqdm(images):
        for tag_name in tags:
            tag_id = tag_manager.get_tag_id(tag_name, hidden=False)
            client.unassign_tag(id, tag_id)
