# DeepNextcloud

NextCloud x DeepDanbooru

<https://twitter.com/faa0311/status/1794672340086964492>

```.env
# .env
DEEPNEXTCLOUD_URL="https://example.com" # NextCloud URL
DEEPNEXTCLOUD_USERNAME="" # NextCloud Username
DEEPNEXTCLOUD_PASSWORD="" # NextCloud Password
DEEPNEXTCLOUD_PATH="illustrations" # NextCloud Path
```

```sh
python -V
# Python 3.10.12
```

```sh
# install pytorch for CUDA 12.1
# https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio
# install other requirements
pip install -r requirements.txt
```


```sh
# Run
python main.py
# en: Remove all tags
python remove_all_tag.py
# use fa0311/twitter-snap
python twitter_snap_normalize.py
```

```sh
# systemd
systemctl link ./deep-nextcloud.timer
systemctl enable deep-nextcloud.timer

systemctl link ./deep-nextcloud.service
systemctl enable deep-nextcloud.service

systemctl start deep-nextcloud.service
systemctl start deep-nextcloud.timer
```