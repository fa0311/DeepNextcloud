# DeepNextcloud

NextCloud x DeepDanbooru

Works with CPU.
<https://huggingface.co/skytnt/deepdanbooru_onnx>


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
pip install -r requirements.txt
python main.py
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