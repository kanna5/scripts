#!/usr/bin/python3

from abc import ABC, abstractmethod
import base64
from functools import cache
import logging
import os
import shutil
import subprocess
import sys

import openai


@cache
def find_api_key() -> str:
    env_key = "OPENAI_APIKEY"
    if env_key in os.environ:
        return os.environ[env_key]

    with open(os.path.expanduser("~/.config/openai/env"), "r", encoding="utf-8") as fd:
        for line in fd:
            line = line.strip()
            if line.startswith(f"{env_key}="):
                return line[len(f"{env_key}=") :]

    raise RuntimeError("Could not find OpenAI API key.")


@cache
def get_blank_audio() -> bytes:
    return base64.b64decode(
        "T2dnUwACAAAAAAAAAAA1tv9CAAAAADAPqBsBE09wdXNIZWFkAQE4AYC7AAAAAABPZ2dTAAAAAAAAAAAAADW2/0IBAA"
        "AANyzVsgGET3B1c1RhZ3MMAAAATGF2ZjYxLjcuMTAwAwAAACMAAABFTkNPREVSPW9wdXNlbmMgZnJvbSBvcHVzLXRv"
        "b2xzIDAuMgsAAAB0aXRsZT1lbXB0eS4AAABFTkNPREVSX09QVElPTlM9LS1jb21wIDEwIC0tYml0cmF0ZSA2IC0tc3"
        "BlZWNoT2dnUwAEuCYAAAAAAAA1tv9CAgAAABZRKOcLBwYGBgYGBgYGBgYIC+Y7I6tgCAissw7GCAissw7GCAissw7G"
        "CAissw7GCAissw7GCAissw7GCAissw7GCAissw7GCAissw7GCAissw7G"
    )


class OutputHandler(ABC):
    @abstractmethod
    def handle(self, data: bytes):
        pass

    @abstractmethod
    def close(self):
        pass


class StdoutOutputHandler(OutputHandler):
    def handle(self, data: bytes):
        sys.stdout.buffer.write(data)

    def close(self):
        pass


class PlayerOutputHandler(OutputHandler):
    MPV_OPTS = ["--input-terminal=no", "--audio-display=no", "--quiet"]

    def __init__(self):
        player_path = shutil.which("mpv")
        if player_path is None:
            raise RuntimeError("Could not find `play` from the sox package")
        self._player_path = player_path
        self._proc = None

    def _init_audio(self):
        """Play a short, blank audio clip to initialize the audio device and avoid clipping the
        start of the speech."""
        audio = get_blank_audio()
        subprocess.run(
            [self._player_path, *self.MPV_OPTS, "-"],
            input=audio,
            capture_output=True,
            check=True,
        )

    def handle(self, data: bytes):
        if self._proc is None:
            self._init_audio()
            self._proc = subprocess.Popen(
                [self._player_path, *self.MPV_OPTS, "-"], stdin=subprocess.PIPE
            )

        assert self._proc.stdin is not None
        self._proc.stdin.write(data)
        self._proc.stdin.flush()

    def close(self):
        if self._proc is not None:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            self._proc.wait()


def main():
    try:
        client = openai.OpenAI(api_key=find_api_key())
    except Exception as err:
        raise RuntimeError("Failed to create OpenAI client") from err

    if os.isatty(sys.stdin.fileno()):
        logging.info(
            "Type in the speech content ending with a newline, then press Ctrl-D to finish."
        )

    content = sys.stdin.read().strip()
    if len(content) == 0:
        logging.fatal("Speech content was empty.")

    logging.info("Sending request")
    with client.audio.speech.with_streaming_response.create(
        model="tts-1-hd", voice="nova", input=content, response_format="opus"
    ) as resp:
        if os.isatty(sys.stdout.fileno()):
            handler = PlayerOutputHandler()
        else:
            logging.info("Writing audio file to stdout")
            handler = StdoutOutputHandler()

        try:
            for chunk in resp.iter_bytes():
                handler.handle(chunk)
        finally:
            handler.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
