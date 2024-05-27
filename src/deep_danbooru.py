import numpy as np
import torch
import tqdm
from PIL import Image

from TorchDeepDanbooru.deep_danbooru_model import DeepDanbooruModel


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
        tags = list(probability_dict.keys())

        rating_name, rating_score = max(rating_dict.items(), key=lambda x: x[1])
        if rating_score < score_threshold / 2:
            return tags, None
        else:
            return tags, rating_name
