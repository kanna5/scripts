#!/usr/bin/python3
"""
A simple interactive client for OpenAI's GPT chat API.

API Key is read from the environment variable `OPENAI_APIKEY`, or from `{{config_dir}}/openai/env` if it's not set.
Templates are loaded from `{{config_dir}}/openai/chat_templates.yml`.
The value of `{{config_dir}}` is platform-dependent. On Linux it's `$HOME/.config`.
"""

import argparse
from collections.abc import Generator
from io import StringIO
import json
import os
import platform
import readline
import shutil
import subprocess
import sys
import threading
from typing import Optional, OrderedDict

import requests
import yaml

COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_RED = "\033[31m"
COLOR_YELLOW = "\033[33m"
COLOR_PURPLE = "\033[35m"
COLOR_CYAN = "\033[36m"
COLOR_GRAY = "\033[90m"  # aka. bright black

ASSISTANT_NAME = "Patchouli"
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.42

SYSTEM_MSG = f"""
{ASSISTANT_NAME} is a helpful assistant.
{ASSISTANT_NAME} must answer questions as truthfully as possible. If the user's intention is unclear, {ASSISTANT_NAME} may ask for more context.
{ASSISTANT_NAME} must use a lot of different emojis in chat 😊.
""".strip()

settings = {
    "debug": False,
    "quiet": False,
    "silent": False,
    "no_color": False,
    "template": None,
    "keep_messages": None,
    "pin_first": None,
}


def get_confdir() -> str:
    """WARN: not tested"""
    system = platform.system()
    if system == "Windows":
        return os.path.join(os.environ["APPDATA"], "openai")
    if system == "Darwin":
        return os.path.expanduser("~/Library/Application Support/openai")
    if "XDG_CONFIG_HOME" in os.environ:
        return os.path.join(os.environ["XDG_CONFIG_HOME"], "openai")

    return os.path.expanduser("~/.config/openai")


def get_apikey() -> str:
    if "OPENAI_APIKEY" in os.environ:
        return os.environ["OPENAI_APIKEY"]

    cfg_path = os.path.join(get_confdir(), "env")
    with open(cfg_path, "r", encoding="utf-8") as fd:
        for line in fd:
            parts = line.split("=", 1)
            if len(parts) < 2:
                continue
            if parts[0].strip() == "OPENAI_APIKEY":
                return parts[1].strip()

    raise KeyError("token not defined")


class FzfError(RuntimeError):
    pass


class Message:
    role: str
    content: str
    pin: bool

    __slots__ = ("role", "content", "pin")

    def __init__(self, role: str, content: str, pin: bool = False) -> None:
        self.role = role
        self.content = content
        self.pin = pin

    @staticmethod
    def system(content: str, pin: bool = True) -> "Message":
        return Message("system", content, pin)

    @staticmethod
    def user(content: str, pin: bool = False) -> "Message":
        return Message("user", content, pin)

    @staticmethod
    def assistant(content: str, pin: bool = False) -> "Message":
        return Message("assistant", content, pin)


class Conversation:
    """A conversation with the assistant.
    TODO: count token usage and rotate messages when limit reached (maybe)
    """

    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    messages: list[Message]
    keep_messages: int = 0
    """How many messages to keep in the conversation.
    0 means unlimited.
    1 means only keep the last message.

    Since previous messages will be sent as context and will be billed, you may want to limit the number of messages to
    keep, and use pinned messages for necessary instructions.
    Pinned messages are not affected by this setting and are not counted towards the limit.
    """

    pin_first: int = 0
    """If set, first {pin_first} added messages will be pinned."""

    _msgcnt: int = 0

    def __init__(self, **kwargs) -> None:
        if "model" in kwargs:
            self.model = kwargs["model"]
            assert isinstance(self.model, str)

        if "temperature" in kwargs:
            self.temperature = kwargs["temperature"]
            assert isinstance(self.temperature, float)

        if "keep_messages" in kwargs:
            self.keep_messages = kwargs["keep_messages"]
            assert isinstance(self.keep_messages, int)

        if "pin_first" in kwargs:
            self.pin_first = kwargs["pin_first"]
            assert isinstance(self.pin_first, int)

        self.messages = []
        if "messages" in kwargs:
            msgs = kwargs["messages"]
            assert isinstance(msgs, list)
            for msg in msgs:
                assert isinstance(msg, Message)
                self.messages.append(msg)

    def add_message(self, msg: Message):
        if self._msgcnt < self.pin_first:
            msg.pin = True
        self.messages.append(msg)
        self._msgcnt += 1
        return self

    def add_user(self, msg: str) -> "Conversation":
        return self.add_message(Message.user(msg))

    def add_assistant(self, msg: str) -> "Conversation":
        return self.add_message(Message.assistant(msg))

    def to_request(self):
        req_msgs = []
        cnt = 0
        for msg in reversed(self.messages):
            if msg.pin:
                req_msgs.append(msg)
                continue
            if self.keep_messages <= 0 or cnt < self.keep_messages:
                req_msgs.append(msg)
                cnt += 1

        return {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                }
                for m in reversed(req_msgs)
            ],
            "stream": True,
        }


class Template:
    def __init__(self, data: Optional[dict] = None) -> None:
        data = data or {}
        self._name = data["name"] if "name" in data else ""
        self._description = data["description"] if "description" in data else ""
        self._template = data["template"] if "template" in data else {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def to_conversation(self, vars_: Optional[dict] = None, overrides: Optional[dict] = None):
        vars_ = vars_ or {}  # Fill template variables in the messages. Not implemented yet. TODO
        tpl = {}
        tpl.update(self._template)
        tpl.update(overrides or {})
        if "messages" in tpl:
            tpl["messages"] = [Message(**msg) for msg in tpl["messages"]]

        return Conversation(**tpl)


class TplMan:
    """Template manager"""

    def __init__(self, tpl_path: str = "") -> None:
        if not tpl_path:
            tpl_path = os.path.join(get_confdir(), "chat_templates.yml")

        tpls_raw = []
        if os.path.isfile(tpl_path):
            with open(tpl_path, "r", encoding="utf-8") as fd:
                tpls_raw = yaml.safe_load(fd)

        self._tpls = OrderedDict()
        for tpl in tpls_raw:
            tpl_instance = Template(tpl)
            self._tpls[tpl_instance.name] = tpl_instance

        if "assistant" not in self._tpls:
            self._tpls["assistant"] = self._get_default()
            self._tpls.move_to_end("assistant", last=False)

    def _get_default(self) -> Template:
        return Template(
            {
                "name": "assistant",
                "description": "A conversation with assistant",
                "template": {
                    "model": DEFAULT_MODEL,
                    "temperature": DEFAULT_TEMPERATURE,
                    "keep_messages": 0,
                    "pin_first": 0,
                    "messages": [{"role": "system", "pin": True, "content": SYSTEM_MSG}],
                },
            }
        )

    def list(self) -> list[tuple[str, str]]:
        return [(k, v.description) for k, v in self._tpls.items()]

    def get(self, name: str) -> Template:
        return self._tpls[name]

    def fuzzy_find(self, query: Optional[str] = None) -> Template:
        fzf_exe = shutil.which("fzf")
        if not fzf_exe:
            raise FzfError("fzf is not installed")

        fzf_input_lines = "".join([f"{k}|[{k}]: {v.description}\n" for k, v in self._tpls.items()])

        fzf_cmd = [
            fzf_exe,
            "--delimiter=\\|",
            "--with-nth",
            "2..",
            "--height=~90%",
            "--reverse",
            "--prompt=Pick a template: ",
        ]
        if query:
            fzf_cmd.append(f"--query={query}")

        try:
            fzf_proc = subprocess.run(
                fzf_cmd, input=fzf_input_lines, stdout=subprocess.PIPE, text=True, check=True
            )
        except subprocess.CalledProcessError as err:
            raise FzfError("no template selected") from err

        parts = fzf_proc.stdout.split("|", 1)
        return self.get(parts[0])


class Client:
    def __init__(self, apikey: str, api_base: str = "https://api.openai.com") -> None:
        self._api_base = api_base
        self._apikey = apikey
        self._session = requests.Session()

    def complete(self, conversation: Conversation) -> Generator[str, None, None]:
        req = conversation.to_request()
        if settings["debug"]:
            print(
                in_color(COLOR_GRAY, json.dumps(req, indent=2, ensure_ascii=False)), file=sys.stderr
            )

        output_buffer = StringIO()

        with self._session.post(
            f"{self._api_base}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._apikey}"},
            json=req,
            stream=True,
            timeout=(3, 10),
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

    def connect(self) -> None:
        """Establish a connection with the API server in advance for faster response"""
        try:
            _ = self._session.options(f"{self._api_base}/v1/chat/completions", timeout=(3, 5))
        except requests.RequestException:
            pass


def in_color(color: str, message: str, bold: bool = False) -> str:
    if settings["no_color"]:
        return message

    bold_ctrl = COLOR_BOLD if bold else ""
    return f"{bold_ctrl}{color}{message}{COLOR_RESET}"


def interactive(client: Client, conv: Conversation) -> None:
    threading.Thread(target=client.connect, daemon=True).start()
    print_hint = not settings["silent"] and not settings["quiet"]

    if print_hint:
        print("Hint: End your message with a newline and press Ctrl+D to send it.", file=sys.stderr)
        print("Hint: Press Ctrl+D without any input to exit.", file=sys.stderr)
    try:
        readline.parse_and_bind("set editing-mode vi")
        # To allow pasting content that contains tab characters.
        readline.set_completer(lambda text, state: text + "\t" if state == 0 else None)
        if print_hint:
            print("Hint: VI editing mode enabled", file=sys.stderr)
    except Exception:
        print("WARN: Failed to set vi editing mode", file=sys.stderr)

    while True:
        print(in_color(COLOR_CYAN, "\nUSER:"))
        input_buf = StringIO()
        try:
            while True:
                line = input()
                print(line, file=input_buf)
                if not line:
                    print("")
        except EOFError:
            pass
        except KeyboardInterrupt:
            return

        user_msg = input_buf.getvalue().strip()
        if not user_msg:
            break

        conv.add_user(user_msg)

        print(in_color(COLOR_PURPLE, "\nASSISTANT:"))
        for content in client.complete(conv):
            print(content, end="")
            sys.stdout.flush()

        print("")


def one_shot(client: Client, conv: Conversation):
    user_msg = sys.stdin.read().strip()
    print(in_color(COLOR_CYAN, "USER:"))
    print(user_msg)
    print(in_color(COLOR_PURPLE, "\nASSISTANT:"))

    for content in client.complete(conv.add_user(user_msg)):
        print(content, end="")
        sys.stdout.flush()

    print("")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="do not print hints")
    parser.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="do not print anything except the response from assistant. Implies -q",
    )
    parser.add_argument(
        "-k",
        "--keep-messages",
        type=int,
        help="number of messages to keep in the conversation, 0 means no limit, 1 means only keep the last message. Overrides the value in the template.",
    )
    parser.add_argument(
        "-p",
        "--pin-first",
        type=int,
        help="number of messages to pin from the beginning of the conversation. Overrides the value in the template.",
    )
    parser.add_argument(
        "-t",
        "--pick-template",
        action="store_true",
        help="pick a template (with fzf if installed)",
    )
    parser.add_argument("-l", "--list", action="store_true", help="list templates")
    parser.add_argument(
        "template",
        nargs="?",
        help='name of the template to use. default is "assistant".',
    )
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    parser.add_argument("--no-color", action="store_true", help="disable color output.")
    return parser.parse_args()


def main():
    args = parse_args()
    settings.update(vars(args))

    if not sys.stdout.isatty():
        settings["no_color"] = True

    tplm = TplMan()
    if args.list:
        for name, description in tplm.list():
            print(f"{in_color(COLOR_PURPLE, name, bold=True)}: {description}")
        return 0

    tpl = tplm.get("assistant")  # Default template
    if args.pick_template:
        try:
            tpl = tplm.fuzzy_find(args.template)
            print(in_color(COLOR_GRAY, f"Picked template: {tpl.name}"))
        except (FzfError, KeyError) as err:
            print(in_color(COLOR_YELLOW, str(err)), file=sys.stderr)
            return 1

    elif args.template:
        try:
            tpl = tplm.get(args.template)
        except KeyError:
            print(in_color(COLOR_YELLOW, f"Template {args.template!r} not found."), file=sys.stderr)
            return 1

    conv = tpl.to_conversation()
    if args.pin_first is not None:
        conv.pin_first = args.pin_first
    if args.keep_messages is not None:
        conv.keep_messages = args.keep_messages

    try:
        client = Client(get_apikey())
    except Exception as err:
        print(in_color(COLOR_RED, f"Failed to initialize client: {err}"), file=sys.stderr)
        return 1

    try:
        if sys.stdin.isatty():
            interactive(client, conv)
        else:
            one_shot(client, conv)
    except requests.HTTPError as err:
        print(in_color(COLOR_RED, "Error:" + str(err)), file=sys.stderr)
        print(err.response.text, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
