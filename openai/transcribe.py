#!/usr/bin/python3

import argparse
import json
import logging
import os
import sys
import typing
from typing import Optional

import openai

logger = logging.getLogger(__name__)


# Run whisper --help to get the list
VALID_LANGUAGES = set(
    "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he"
    "hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne"
    "nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk"
    "ur,uz,vi,yi,yo,zh".split(",")
)

VALID_RESPONSE_FORMATS = set(["json", "text", "srt", "verbose_json", "vtt"])


class AuthError(RuntimeError):
    pass


def get_auth_info() -> str:
    envkey = "OPENAI_APIKEY"
    if envkey in os.environ and os.environ[envkey]:
        return os.environ[envkey]

    with open(os.path.expanduser("~/.config/openai/env")) as fd:
        for line in fd:
            parts = line.strip().split("=", 1)
            if parts[0].strip() == envkey:
                return parts[1].strip()

    raise AuthError("no api key specified")


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe audio with OpenAI's Whisper API")
    parser.add_argument(
        "-l",
        "--language",
        help="Language code. Optional.",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        help="Temperature. Default is 0.2.",
        default=0.2,
        type=float,
    )
    parser.add_argument(
        "-p",
        "--prompt",
        help="Prompt. Default is empty.",
    )
    parser.add_argument(
        "-f",
        "--response-format",
        default="vtt",
        help="Supported formats: json, text, srt, verbose_json, vtt. Default is vtt.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override if output files already exist.",
    )
    parser.add_argument(
        "audio_file",
        nargs="+",
        help="Audio files. Must not exceed 20MiB in size.",
    )
    return parser.parse_args()


def transcribe(
    client: openai.Client,
    audio_file: typing.IO,
    language: Optional[str] = None,
    temperature: Optional[float] = None,
    prompt: Optional[str] = None,
    response_format: Optional[str] = None,
):
    api_args = {}

    if language is not None:
        assert language in VALID_LANGUAGES
        api_args["language"] = language
    if temperature is not None:
        assert 0 <= temperature <= 1
        api_args["temperature"] = temperature
    if response_format is not None:
        assert response_format in VALID_RESPONSE_FORMATS
        api_args["response_format"] = response_format
    if prompt is not None:
        api_args["prompt"] = prompt

    return client.audio.transcriptions.create(model="whisper-1", file=audio_file, **api_args)


def get_output_filename(input_filename: str, format: str):
    suffix_map = {
        "json": "json",
        "text": "txt",
        "srt": "srt",
        "verbose_json": "json",
        "vtt": "vtt",
    }
    if format in suffix_map:
        return input_filename + "." + suffix_map[format]

    raise KeyError(f"invalid format: {format}")


def write_result(output_file: typing.IO, format: str, result):
    if format in ["json", "verbose_json"]:
        output_file.write(json.dumps(result))
    elif format == "text":
        output_file.write(result.text)
    elif format in ["srt", "vtt"]:
        output_file.write(f"{result.strip()}\n")

    output_file.flush()


def main():
    args = vars(parse_args())
    client = openai.Client(api_key=get_auth_info())

    for audio_file in args["audio_file"]:
        if not os.path.isfile(audio_file):
            logger.fatal("file not found: %s", audio_file)

        if not args["force"]:
            output_filename = get_output_filename(audio_file, args["response_format"])
            if os.path.exists(output_filename):
                logger.error(
                    "output file already exist: %s. Use --force to override.",
                    output_filename,
                )
                return 1

    for audio_file in args["audio_file"]:
        transcribe_args = {
            k: args[k] for k in ["language", "temperature", "prompt", "response_format"]
        }

        logger.info("Transcribing %r ...", audio_file)
        with open(audio_file, "rb") as fd:
            transcribe_args["audio_file"] = fd
            result = transcribe(client, **transcribe_args)

        output_filename = get_output_filename(audio_file, args["response_format"])
        logger.info(f"Writing to {output_filename!r}.")
        with open(output_filename, "w") as fd:
            write_result(fd, args["response_format"], result)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    )
    sys.exit(main())
