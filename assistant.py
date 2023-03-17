#!/usr/bin/python3
"""
A simple interactive client for OpenAI's GPT chat API.
"""

from collections.abc import Generator
import os
import sys
import json
from io import StringIO

import requests

COLOR_CYAN = "\033[36m"
COLOR_PURPLE = "\033[35m"
COLOR_RESET = "\033[0m"

ASSISTANT_NAME = "Patchouli"

SYSTEM_MSG = f"""
{ASSISTANT_NAME} is a helpful assistant.
"""


def get_apikey() -> str:
    if "OPENAI_APIKEY" in os.environ:
        return os.environ["OPENAI_APIKEY"]

    cfg_path = os.path.expanduser("~/.config/openai/env")
    if "XDG_CONFIG_HOME" in os.environ:
        cfg_path = os.environ["XDG_CONFIG_HOME"] + "/openai/env"

    with open(cfg_path, "r", encoding="utf-8") as fd:
        for line in fd:
            parts = line.split("=", 1)
            if len(parts) < 2:
                continue
            if parts[0].strip() == "OPENAI_APIKEY":
                return parts[1].strip()

    return ""


class Conversation:
    """
    A conversation with the assistant.

    TODO: count token usage and rotate messages when limit reached (maybe)
    TODO: load conversation presets from file (maybe)
    """

    def __init__(self) -> None:
        self.model = "gpt-3.5-turbo"
        self.temperature = 0.42
        self.messages = [{"role": "system", "content": SYSTEM_MSG}]

    def add_user(self, msg: str) -> "Conversation":
        self.messages.append({"role": "user", "content": msg})
        return self

    def add_assistant(self, msg: str) -> "Conversation":
        self.messages.append({"role": "assistant", "content": msg})
        return self

    def to_request(self):
        return {
            "model": self.model,
            "temperature": self.temperature,
            "messages": self.messages,
            "stream": True,
        }


class Client:
    def __init__(self, apikey: str, api_base: str = "https://api.openai.com") -> None:
        self._api_base = api_base
        self._apikey = apikey
        self._session = requests.Session()

    def complete(self, conversation: Conversation) -> Generator[str, None, None]:
        output_buffer = StringIO()

        with self._session.post(
            f"{self._api_base}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._apikey}"},
            json=conversation.to_request(),
            stream=True,
        ) as resp:
            if resp.status_code != 200:
                _ = resp.text  # read full response
                resp.raise_for_status()

            # The model sometimes sends a few empty lines before the actual response
            content_began = False

            for line in resp.iter_lines():
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue

                data = line[6:]
                if data.startswith("[DONE]"):
                    break
                try:
                    decoded = json.loads(line[5:])
                    content = decoded["choices"][0]["delta"]["content"]
                    if not content_began:
                        if not content.strip():
                            continue
                    content_began = True

                    output_buffer.write(content)
                    yield content

                except KeyError:
                    continue

        conversation.add_assistant(output_buffer.getvalue())


def in_color(color: str, message: str) -> str:
    return f"{color}{message}{COLOR_RESET}"


def interactive():
    print("Hint: End your message with a newline and press Ctrl+D to send it.")
    print("Hint: Press Ctrl+D without any input to exit.")

    client = Client(get_apikey())
    conv = Conversation()
    while True:
        print(in_color(COLOR_CYAN, "\nUSER:"))
        input_buf = StringIO()
        try:
            while True:
                print(input(), file=input_buf)
        except EOFError:
            pass

        user_msg = input_buf.getvalue().strip()
        if not user_msg:
            break

        conv.add_user(user_msg)

        print(in_color(COLOR_PURPLE, "\nASSISTANT:"))
        for content in client.complete(conv):
            print(content, end="")
            sys.stdout.flush()

        print("")


def one_shot():
    client = Client(get_apikey())

    user_msg = sys.stdin.read().strip()
    print(in_color(COLOR_CYAN, "USER:"))
    print(user_msg)
    print(in_color(COLOR_PURPLE, "\nASSISTANT:"))

    for content in client.complete(Conversation().add_user(user_msg)):
        print(content, end="")
        sys.stdout.flush()

    print("")


def main():
    if sys.stdin.isatty():
        interactive()
    else:
        one_shot()


if __name__ == "__main__":
    sys.exit(main())
