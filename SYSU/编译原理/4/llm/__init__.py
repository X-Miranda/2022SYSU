import uuid
import xml.etree.ElementTree as ET

from openai import OpenAI
from typing import Callable, List, Literal


def remove_deepseek_r1_think(s: str) -> str:
    if "</think>" in s:
        s = s.split("</think>", 1)[1].strip()
    return s


def remove_md_block_marker(markder: str):
    def remove(s: str) -> str:
        if s.startswith(f"```{markder}"):
            lines = s.splitlines()
            s = "\n".join(lines[1:-1]).strip()
        return s

    return remove


def extract_text_from_xml(tag: str):
    def extract(s: str) -> str:
        et = ET.fromstring(s)
        node = et.find(tag)
        assert node is not None, f"无法找到 tag 为 {tag} 的节点"
        text = node.text
        assert text is not None, "节点中不含有 text！"
        return text.strip()

    return extract


class LLMHelperImpl:
    _instances = {}
    _instances_init = {}

    def __new__(cls, api_key: str, base_url: str):
        key = (api_key, base_url)
        if key not in cls._instances:
            instance = super(LLMHelperImpl, cls).__new__(cls)
            cls._instances[key] = instance
            cls._instances_init[key] = False
        return cls._instances[key]

    def __init__(self, api_key: str, base_url: str) -> None:
        if not LLMHelperImpl._instances_init[(api_key, base_url)]:
            self.__client = OpenAI(api_key=api_key, base_url=base_url)
            self.__sessions = {}
            LLMHelperImpl._instances_init[(api_key, base_url)] = True

    def create_new_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.__sessions[session_id] = []
        return session_id

    def delete_session(self, session_id: str):
        del self.__sessions[session_id]

    def add_content(
        self,
        session_id: str,
        role: Literal["user", "system", "assistant"],
        content: str,
    ) -> None:
        self.__sessions[session_id].append({"role": role, "content": content})

    def chat(
        self,
        session_id: str,
        model: str,
        handlers: List[Callable[[str], str]] = [],
        **params,
    ) -> str:
        messages = self.__sessions[session_id]
        response = (
            self.__client.chat.completions.create(
                messages=messages, model=model, **params
            )
            .choices[0]
            .message.content
        )

        for handler in handlers:
            response = handler(response)
        return response
