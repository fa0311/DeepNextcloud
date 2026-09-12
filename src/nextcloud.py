import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter


class NextCloud:
    def __init__(self, url, username, password, concurrency=8):
        self.session = requests.Session()
        self.username = username
        self.password = password
        self.session.auth = (username, password)
        self.session.mount(url, HTTPAdapter(pool_maxsize=concurrency))
        self.executor = ThreadPoolExecutor(max_workers=concurrency)
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

    def path_list(self, path) -> list[tuple]:
        res = self.request(
            "PROPFIND",
            f"{self.url}/remote.php/dav/files/{self.username}/{path}",
            [
                "d:getlastmodified",
                "d:getcontenttype",
                "oc:fileid",
                "d:href",
                "d:displayname",
                "nc:system-tags",
            ],
        )
        return res

    def recursive_path_list(self, path: str) -> list[tuple]:
        children: list[tuple] = []
        images = self.path_list(path)

        for timestamp, content_type, id, image, displayname, tags in images[1:]:
            if image.endswith("/"):
                children.extend(self.recursive_path_list(f"{path}/{displayname}"))

        return [*images[1:], *children]

    def download(self, href):
        response = self.session.request("GET", f"{self.url}{href}")
        response.raise_for_status()
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
        response = self.session.request(
            "POST",
            self.url + "/remote.php/dav/systemtags/",
            json={
                "userVisible": user_visible,
                "userAssignable": user_assignable,
                "canAssign": can_assign,
                "name": name,
            },
        )
        if not response.ok:
            raise Exception(
                f"Failed to create tag {name!r}: HTTP {response.status_code} {response.text[:300]}"
            )

    def assign_tag(self, file_id, tag_id):
        response = self.session.request(
            "PUT",
            self.url + f"/remote.php/dav/systemtags-relations/files/{file_id}/{tag_id}",
        )
        already_assigned = response.status_code == 409
        if not already_assigned:
            response.raise_for_status()
        return response.text

    def assign_tags(self, file_id, tag_ids):
        for _ in self.executor.map(lambda tag_id: self.assign_tag(file_id, tag_id), tag_ids):
            pass

    def unassign_tag(self, file_id, tag_id):
        response = self.session.request(
            "DELETE",
            self.url + f"/remote.php/dav/systemtags-relations/files/{file_id}/{tag_id}",
        )
        return response.text
