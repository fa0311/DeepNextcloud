import xml.etree.ElementTree as ET

import requests


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
        self,
        name,
        user_visible=True,
        user_assignable=True,
        can_assign=True,
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
