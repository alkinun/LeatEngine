"""Terminal chat, HTTP serving, API keys, inference measurements, and a check of the machine.

Engine imports follow device selection, before tinygrad initializes its backend.
"""

import argparse
import contextlib
import ipaddress
import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from leatengine import pull
from leatengine.defaults import OPTIONS, Overrides
from leatengine.files import later_part, model_name, projector
from leatengine.keys import Keys
from leatengine.pull import DEFAULT_QUANT

if TYPE_CHECKING:
    from leatengine.chat import ChatTemplate, Reply
    from leatengine.server import Server
    from leatengine.vision import Image


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="leatengine", description="A minimal GGUF inference engine built on tinygrad."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    pull = commands.add_parser(
        "pull", help="download a model from Hugging Face, as OWNER/NAME:QUANT",
        description="Downloads a repository's GGUF of a quantization, every part of a split one, "
        "and the vision projector beside it, each checked against Hugging Face's SHA-256, into "
        f"{_data() / 'models'}, where run and serve find it by the same name. An interrupted "
        "download resumes. Gated repositories need a token, HF_TOKEN or huggingface-cli's login.",
    )  # fmt: skip
    pull.add_argument(
        "model", help="OWNER/NAME, or OWNER/NAME:QUANT, QUANT a word of the file's name, as "
        f"Q8_0 or UD-Q4_K_XL, by default {DEFAULT_QUANT}, or the file's name whole",
    )  # fmt: skip
    pull.add_argument("--no-mmproj", action="store_true", help="leave out the vision projector")
    pull.add_argument("--dry-run", action="store_true", help="say what it would download")

    run = commands.add_parser("run", help="chat with a model in the terminal")
    run.add_argument("model", type=Path, help="GGUF file, or a model pulled, OWNER/NAME:QUANT")
    run.add_argument(
        "--max-context", type=_positive,
        help="tokens the chat holds at most; by default the model's own, as far as its KV cache "
        "fits in the memory its weights leave",
    )  # fmt: skip
    _cache_type(run)
    run.add_argument("--temperature", type=_option("temperature"), default=0.7)
    run.add_argument(
        "--top-k", type=_option("top_k"), default=0, help="keep the k likeliest tokens; 0 keeps all"
    )
    run.add_argument("--top-p", type=_option("top_p"), default=1.0)
    run.add_argument("--min-p", type=_option("min_p"), default=0.0)
    run.add_argument("--presence-penalty", type=_option("presence_penalty"), default=0.0)
    run.add_argument("--frequency-penalty", type=_option("frequency_penalty"), default=0.0)
    run.add_argument(
        "--repetition-penalty", type=_option("repetition_penalty"), default=1.0,
        help="divides a repeated token's positive logit and multiplies a negative one, of the "
        "prompt's tokens too; 1 changes nothing",
    )  # fmt: skip
    run.add_argument("--system", help="system prompt")
    run.add_argument(
        "--mmproj", type=Path,
        help="a vision encoder's GGUF, for images; by default the projector beside the model of "
        "its name, if any. `/image PATH` attaches an image to the next message",
    )  # fmt: skip
    run.add_argument(
        "--draft", type=Path,
        help="a drafter's GGUF, for speculative decoding, as Gemma 4's assistant for Gemma 4; by "
        "default the model's own MTP layer, if any, in its file or the mtp-*.gguf beside it",
    )  # fmt: skip

    serve = commands.add_parser("serve", help="serve the OpenAI chat completions API")
    serve.add_argument(
        "models", type=Path, nargs="*",
        help="GGUF files, directories of them, or models pulled, OWNER/NAME:QUANT; the first file "
        "loads at start, others on request",
    )  # fmt: skip
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=_port, default=8080)
    serve.add_argument(
        "--max-context", type=_positive,
        help="tokens a sequence holds at most; by default each model's own, as far as the KV cache "
        "of its slots fits in the memory its weights leave",
    )  # fmt: skip
    serve.add_argument(
        "--slots", type=_positive, default=4,
        help="sequences generating at once, each in a slot of the KV cache that keeps its tokens",
    )  # fmt: skip
    _cache_type(serve)
    serve.add_argument(
        "--draft", type=Path,
        help="a drafter's GGUF, for speculative decoding, of the one model; by default each model "
        "drafts with its own MTP layer, if any, in its file or the mtp-*.gguf beside it",
    )  # fmt: skip
    serve.add_argument(
        "--mmproj", type=Path,
        help="a vision encoder's GGUF, of the one model; by default each model takes the "
        "projector beside it of its name, if any",
    )  # fmt: skip
    serve.add_argument(
        "--embed", type=Path,
        help="an embedding model's GGUF, as Qwen3-Embedding's or EmbeddingGemma 2's, loaded "
        "beside the models to answer /v1/embeddings",
    )  # fmt: skip
    serve.add_argument(
        "--keys", type=Path, nargs="?", const=_data() / "keys.json",
        help="API keys made by `leatengine keys`, one of which every request must hold; "
        f"by default {_data() / 'keys.json'}. Needed to serve beyond this machine",
    )  # fmt: skip

    serve.add_argument(
        "--sampling", type=Path, nargs="?", const=_data() / "sampling.json",
        help="a file of JSON of your own sampling, by model id, or \"*\" for every model, as "
        '{"gpt-oss-20b": {"temperature": 0.8}}, over what each model\'s makers recommend; by '
        f"default {_data() / 'sampling.json'}. Read again as it changes",
    )  # fmt: skip

    keys = commands.add_parser("keys", help="make, list and remove leatengine serve's API keys")
    keys.add_argument("action", choices=("add", "list", "remove"))
    keys.add_argument("name", nargs="?", help="the key's, a person's or an app's, as alkin")
    keys.add_argument("--file", type=Path, default=_data() / "keys.json", help="the file of keys")

    speed = commands.add_parser("bench", help="measure prefill and decode speed")
    speed.add_argument("model", type=Path, help="GGUF file, or a model pulled, OWNER/NAME:QUANT")
    speed.add_argument("-p", "--prompt", type=_positive, default=512, help="prompt tokens")
    speed.add_argument("-n", "--generate", type=_positive, default=128, help="generated tokens")
    speed.add_argument("-r", "--reps", type=_positive, default=3)
    speed.add_argument(
        "-s", "--sequences", type=_positive, default=1, help="sequences generating at once"
    )
    _cache_type(speed)
    speed.add_argument(
        "--draft",
        type=Path,
        help="a drafter's GGUF, for speculative decoding: Gemma 4's assistant for Gemma 4, or "
        "for Qwen3.5's MTP layer the model's own file, or the mtp-*.gguf beside it",
    )
    speed.add_argument(
        "--chat", action="store_true",
        help="decode replies to chat prompts in the model's template instead, as a drafter "
        "guesses them in use, for -n tokens each",
    )  # fmt: skip
    speed.add_argument(
        "--recommended", action="store_true",
        help="with --chat, sample the replies as the model's makers recommend, rather than "
        "greedily",
    )  # fmt: skip
    speed.add_argument("--json", action="store_true", help="print one JSON object")
    speed.add_argument(
        "--share", action="store_true",
        help="print one JSON object to share: the result, the run, the model's files by their "
        "SHA-256, the commits and the machine, as docs/benchmarks.md has it, naming no one",
    )  # fmt: skip

    quality = commands.add_parser(
        "perplexity", help="measure quality on text, or against llama.cpp"
    )
    quality.add_argument("model", type=Path, help="GGUF file, or a model pulled, OWNER/NAME:QUANT")
    source = quality.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", type=Path, help="text file to score")
    source.add_argument(
        "--kl-base", type=Path, help="logits from llama-perplexity --kl-divergence-base"
    )
    quality.add_argument(
        "--ctx", type=_positive, default=512,
        help="chunk size, as in llama-perplexity -c; a --kl-base file has its own",
    )  # fmt: skip
    quality.add_argument("--chunks", type=_positive, help="score only the first N chunks")
    _cache_type(quality)
    quality.add_argument(
        "--decode", action="store_true", help="score one token at a time, as generation runs"
    )
    quality.add_argument("--json", action="store_true", help="print one JSON object")

    check = commands.add_parser(
        "doctor", help="check this machine for LeatEngine on Strix Halo: its GPU, memory and clang"
    )
    check.add_argument("--json", action="store_true", help="print one JSON object")

    args = parser.parse_args(argv)
    if args.command == "serve":
        _check_serve(serve, args)
        args.models = [_pulled(path) for path in args.models]
    if args.command in ("run", "bench", "perplexity"):
        args.model = _pulled(args.model)
    if args.command in ("run", "serve", "bench", "perplexity"):
        _device()
    handlers = {
        "pull": _pull, "run": _run, "serve": _serve, "keys": _keys, "bench": _bench,
        "perplexity": _perplexity, "doctor": _doctor,
    }  # fmt: skip
    handlers[args.command](args)


# GPU device files, by tinygrad backend name, in discovery order
_GPUS = (("NV", "/dev/nvidiactl"), ("AMD", "/dev/kfd"))


def _device() -> None:
    # tinygrad's device, where DEV does not say it: NV of an NVIDIA GPU, as leatengine's kernels are
    # written for, rather than CUDA, which tinygrad may choose instead, a load then putting some
    # tensors on each and hanging; or AMD's, as Strix Halo's
    if "DEV" not in os.environ:
        found = next((dev for dev, path in _GPUS if Path(path).exists()), None)
        if found is not None:
            os.environ["DEV"] = found


def _data() -> Path:
    # API keys, sampling overrides and models pulled, in the XDG application data directory
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "leatengine"


def _pulled(path: Path) -> Path:
    # a model's file: as given, or of a model pulled, OWNER/NAME[:QUANT]
    if path.exists():
        return path
    if pull.parse(str(path)) is None:
        raise SystemExit(f"{path}: there is no such file")
    try:
        found = pull.local(_data() / "models", str(path))
    except ValueError as e:  # as of a split model's part not there, or two of its quantization
        raise SystemExit(f"{path}: {e}; leatengine pull {path} again") from None
    if found is None:
        raise SystemExit(f"{path} is no file, nor a model pulled: leatengine pull {path}")
    return found


@contextlib.contextmanager
def _said() -> Iterator[None]:
    # an error of a file or of what was asked of it, as a model's context, said rather than
    # traced back
    try:
        yield
    except (OSError, ValueError, MemoryError) as e:
        raise SystemExit(str(e)) from None


def _pull(args: argparse.Namespace) -> None:
    try:
        path = pull.pull(args.model, _data() / "models", not args.no_mmproj, args.dry_run)
    except (ValueError, OSError) as e:
        raise SystemExit(f"pull failed: {e}") from None
    if not args.dry_run:
        print(f"{path}\nrun it: leatengine run {args.model}")


def _positive(text: str) -> int:
    # an argument that counts something, of which there must be one at least
    if (n := int(text)) < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {n}")
    return n


def _option(name: str) -> Any:
    # a sampling option's argument type, of the bounds a request's and the sampling file's have
    kind, low, high = OPTIONS[name]

    def parse(text: str) -> int | float:
        value = kind(text)
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"must be from {low} to {high}, got {text}")
        return value

    parse.__name__ = kind.__name__  # as argparse names the type in its errors
    return parse


def _cache_type(parser: argparse.ArgumentParser) -> None:
    # how the KV cache holds keys and values, as leatengine.model.CACHE_TYPES, which importing
    # would start tinygrad before the device is chosen
    parser.add_argument(
        "--cache-type", choices=("f16", "q8_0"), default="f16",
        help="what the KV cache holds keys and values as: f16, or q8_0 in about half the memory",
    )  # fmt: skip


def _port(text: str) -> int:
    if not 0 <= (n := int(text)) <= 65535:
        raise argparse.ArgumentTypeError(f"must be from 0 to 65535, got {n}")
    return n


def _check_serve(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    # --keys and --sampling take a file or none, so a model right after either is taken for its
    # file: said, rather than a model's file read as keys
    for flag in ("keys", "sampling"):
        if (path := getattr(args, flag)) is not None and (path.suffix == ".gguf" or path.is_dir()):
            parser.error(f"--{flag} takes its file, not {path}: give the models before it, or "
                         f"its file after it")  # fmt: skip
    if not args.models:
        parser.error("the following arguments are required: models")


def _url(host: str, port: int) -> str:
    # an address to browse to: this machine's, where the server takes every one
    host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    return f"http://{f'[{host}]' if ':' in host else host}:{port}"


def _run(args: argparse.Namespace) -> None:
    from tinygrad import Device

    from leatengine.chat import ChatTemplate, split_reply
    from leatengine.draft import own_drafter
    from leatengine.engine import Engine
    from leatengine.sampler import Sampling
    from leatengine.vision import Image, beside

    vision, draft = args.mmproj or beside(args.model), args.draft or own_drafter(args.model)
    with _said():
        engine = Engine(args.model, max_context=args.max_context, draft=draft, vision=vision,
                        cache_type=args.cache_type)  # fmt: skip
    chat, tok = ChatTemplate(engine.gguf.metadata, engine.tokenizer), engine.tokenizer
    sampling = Sampling(
        args.temperature, args.top_k, args.top_p, args.min_p, args.presence_penalty,
        args.frequency_penalty, args.repetition_penalty,
    )  # fmt: skip
    first = [{"role": "system", "content": args.system}] if args.system else []
    messages, pictures, attached = list(first), list[Image](), list[Image]()
    print(f"{model_name(engine.gguf.path)} on {Device.DEFAULT}: a context of {engine.max_context} "
          f"tokens, {engine.fit}")  # fmt: skip
    print("compiling...", end="", flush=True)
    engine.warm_up()
    print(' ready. /help lists commands; """ begins and ends a message of lines; Ctrl-D quits.')
    while True:
        try:
            line = _read()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        word, _, rest = line.partition(" ")
        if word in ("/quit", "/exit"):
            return
        if word == "/help":
            print(_HELP)
            continue
        if word == "/reset":  # the chat starts over, of its system prompt alone
            messages, pictures, attached = list(first), [], []
            print("[the chat starts over]")
            continue
        if word == "/system":  # of the chat so far, the system prompt replaced, or with none taken
            first = [{"role": "system", "content": rest.strip()}] if rest.strip() else []
            messages = first + [m for m in messages if m["role"] != "system"]
            print(f"[the system prompt {'is set' if first else 'is gone'}]")
            continue
        if word == "/image":
            try:  # shown before the next message's text
                path = Path(rest.strip().strip("'\"")).expanduser()  # as terminals quote one
                attached.append(engine.image(path.read_bytes()))
            except (OSError, ValueError) as e:  # as of a model without a vision encoder
                print(f"[no image: {e}]")
            continue
        if line.startswith("//"):  # a message that begins with a slash, as /usr's
            line = line[1:]
        elif line.startswith("/"):
            print(f"[{word} is no command: /help lists them; // begins a message of a slash]")
            continue
        n = len(attached)  # the message's images, before its text
        images = [{"type": "image"}] * n + [{"type": "text", "text": line}]
        messages.append({"role": "user", "content": images if n else line})
        pictures, attached = pictures + attached, []
        try:
            prompt, thinking = _prompt(chat, messages, pictures)
            if len(prompt) >= engine.max_context and len(messages) > len(first) + 1:
                # from the message just written, if it fits alone; else the chat as it was
                alone, shown_alone = [*first, messages[-1]], pictures[len(pictures) - n :]
                held, (prompt, thinking) = len(prompt), _prompt(chat, alone, shown_alone)
                if len(prompt) < engine.max_context:
                    print(f"[the conversation is {held} tokens, over --max-context; starting "
                          "over]")  # fmt: skip
                    messages, pictures = alone, shown_alone
        except ValueError as e:  # as Mistral 7B v0.3's template refuses a system prompt
            print(f"[{e}; the message is dropped{', as /system takes it away' if first else ''}]")
            messages.pop()
            pictures = pictures[: len(pictures) - n]
            continue
        if len(prompt) >= engine.max_context:
            print(f"[the message makes {len(prompt)} tokens, over --max-context]")
            messages.pop()
            pictures = pictures[: len(pictures) - n]
            continue
        reply, step, start = [], tok.stream(), time.perf_counter()
        text, shown = "", ("", "")  # the reply so far, and its reasoning and text printed
        try:
            for t in engine.generate(prompt, engine.max_context - len(prompt), sampling,
                                     images=pictures):  # fmt: skip
                if t in tok.eog_ids:
                    break
                reply.append(t)
                text += step(t)
                shown = _show(split_reply(text, chat.form, thinking), shown)
        except KeyboardInterrupt:
            pass
        text += step(None)
        parts = split_reply(text, chat.form, thinking, done=True)
        _show(parts, shown)
        rate = len(reply) / (time.perf_counter() - start)
        held = len(prompt) + len(reply)
        print(f"\n\033[2m[{len(reply)} tokens, {rate:.1f} tok/s; the chat holds {held} of "
              f"{engine.max_context} tokens]\033[0m")  # fmt: skip
        messages.append({"role": "assistant", "content": parts.content})


def _prompt(
    chat: "ChatTemplate", messages: list[dict], images: list["Image"]
) -> tuple[list[int], bool]:
    # the chat's prompt, showing its images in turn, and whether it opens a block of reasoning
    import jinja2

    try:
        rendered = chat.render(messages)
    except jinja2.TemplateError as e:  # such as a system prompt the template does not take
        raise ValueError(f"the model's chat template refuses this chat: {e}") from None
    return chat.tokens(rendered, [image.tokens for image in images]), chat.opens_thinking(rendered)


_HELP = "\n".join([
    "/system TEXT  set the system prompt, of the chat so far too; /system alone takes it away",
    "/reset        start the chat over, of its system prompt alone",
    "/image PATH   attach an image to the next message, of a model that takes them",
    "/help         this",
    "/quit         quit, as Ctrl-D does",
    '"""           begin a message of several lines, which another """ ends',
    "//            begin a message that begins with a slash",
])  # fmt: skip


def _read() -> str:
    # a message: a line, or the lines between two of three quotes, as Ollama's
    line = input("> ")
    if not line.startswith('"""'):
        return line
    if len(line) > 5 and line.endswith('"""'):
        return line[3:-3]
    lines = [line[3:]] if line[3:] else []
    while not (more := input(". ")).endswith('"""'):
        lines.append(more)
    return "\n".join(lines + ([more[:-3]] if more[:-3] else []))


def _show(parts: "Reply", shown: tuple[str, str]) -> tuple[str, str]:
    # prints what is new of a reply's reasoning, dimmed, and of its text
    reasoning, content = shown
    if len(parts.reasoning) > len(reasoning):
        print(f"\033[2m{parts.reasoning[len(reasoning) :]}\033[0m", end="", flush=True)
    if len(parts.content) > len(content):
        if not content and parts.reasoning:
            print()
        print(parts.content[len(content) :], end="", flush=True)
    return parts.reasoning, parts.content


def _serve(args: argparse.Namespace) -> None:
    from tinygrad import Device

    from leatengine.draft import mtp
    from leatengine.server import Server

    models = [f for p in args.models for f in (_found(p) if p.is_dir() else [p])]
    # each model's vision encoder and MTP layer, not models
    models = [f for f in models if not projector(f) and not mtp(f)]
    if not models:
        raise SystemExit("no GGUF files: give some, or directories that hold some")
    for flag in ("draft", "mmproj"):
        if getattr(args, flag) and len(models) > 1:
            raise SystemExit(f"--{flag} is of one model: serve that one alone")
    if args.keys is None and not _loopback(args.host):
        raise SystemExit(f"to serve beyond this machine, at {args.host}, give --keys: make one "
                         "with `leatengine keys add NAME`")  # fmt: skip
    if args.keys is not None and not args.keys.exists():
        raise SystemExit(f"there are no keys at {args.keys}: make one with "
                         f"`leatengine keys add NAME --file {args.keys}`")  # fmt: skip
    options = {"max_context": args.max_context, "slots": args.slots, "draft": args.draft,
               "vision": args.mmproj, "cache_type": args.cache_type}  # fmt: skip
    keys = Keys(args.keys) if args.keys is not None else None
    sampling = Overrides(args.sampling) if args.sampling is not None else None
    if sampling is not None:
        try:
            sampling.of("")
        except (OSError, ValueError) as e:
            raise SystemExit(f"the sampling at {args.sampling} cannot be read: {e}") from e
    with Server(
        models, args.host, args.port, keys, sampling, embed=args.embed, **options
    ) as server:
        # served at once, the model loading meanwhile, so that a client asking while it compiles
        # hears it is loading, its completions waiting for it, rather than no answer
        url, name = _url(args.host, server.server_port), model_name(models[0])
        failed: list[Exception] = []
        print(f"{name} on {Device.DEFAULT}, the API at {url}/v1, compiling...", flush=True)
        threading.Thread(target=_start, args=(server, name, failed), daemon=True).start()
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()
        if failed:
            raise failed[0]


def _found(directory: Path) -> list[Path]:
    # the GGUFs of a directory of models that run here, saying which do not, as drafters of other
    # engines' kinds beside a model
    from leatengine.gguf import GGUF
    from leatengine.model import supported

    found = []
    for path in sorted(directory.glob("*.gguf")):
        if later_part(path):  # of a split GGUF, which its first part opens
            continue
        try:
            arch = GGUF.open(path).metadata.get("general.architecture", "")
        except (OSError, ValueError) as e:  # not a GGUF after all, or truncated
            print(f"skipping {path.name}: {e}", file=sys.stderr)
            continue
        if supported(arch):
            found.append(path)
        else:
            print(f"skipping {path.name}: its architecture, {arch!r}, is not supported",
                  file=sys.stderr)  # fmt: skip
    return found


def _loopback(host: str) -> bool:
    # whether a host the server binds to is this machine's alone
    with contextlib.suppress(ValueError):
        return ipaddress.ip_address(host).is_loopback
    return host == "localhost"


def _keys(args: argparse.Namespace) -> None:
    keys = Keys(args.file)
    if args.action != "list" and not args.name:
        raise SystemExit(f"leatengine keys {args.action} needs the key's name")
    try:
        if args.action == "list":
            for listed in keys.listed():
                print(f"{listed['name']}\t{listed['created']}")
        elif args.action == "add":
            key = keys.add(args.name)
            print(f"{key}\n\nThe key of {args.name}, shown this once: keep it where its app reads "
                  f"it, as OPENAI_API_KEY. {args.file} keeps its hash alone.")  # fmt: skip
        else:
            keys.remove(args.name)
            print(
                f"{args.name}'s key is removed: leatengine serve refuses it "
                "from the next request on."
            )
    except (ValueError, LookupError) as e:
        raise SystemExit(str(e)) from e


def _start(server: "Server", name: str, failed: list[Exception]) -> None:
    # loads the model a server starts with, or stops the server if it cannot
    try:
        server.load(name)
        print("ready. Ctrl-C quits.", flush=True)
    except Exception as e:
        failed.append(e)
        server.shutdown()


def _bench(args: argparse.Namespace) -> None:
    from tinygrad import Device

    from leatengine import bench
    from leatengine.defaults import recommended
    from leatengine.engine import Engine
    from leatengine.gguf import GGUF
    from leatengine.model import Config
    from leatengine.sampler import GREEDY, Sampling

    n = args.sequences
    context = (bench.CHAT_CONTEXT if args.chat else args.prompt) + args.generate
    if args.chat:  # as long as the model's context allows
        metadata = GGUF.open(args.model).metadata
        context = min(context, Config.from_gguf(metadata).context_length)
    with _said():
        engine = Engine(
            args.model, max_context=context, prefill_chunk=args.prompt, slots=n,
            draft=args.draft, cache_type=args.cache_type,
        )  # fmt: skip
    name = model_name(engine.gguf.path)
    head = {"model": name, "device": Device.DEFAULT, "cache_type": args.cache_type}
    title = f"{name} on {Device.DEFAULT}, {args.cache_type} KV cache"
    run = {"device": Device.DEFAULT, "cache_type": args.cache_type, "prompt": args.prompt,
           "generate": args.generate, "reps": args.reps, "sequences": n, "chat": args.chat,
           "draft": args.draft.name if args.draft else None}  # fmt: skip
    if args.chat:
        made: dict[str, Any] = dict(recommended(engine.gguf.metadata)[0])  # the reasoning set's
        sampling = Sampling(**made) if args.recommended else GREEDY
        chat = bench.chat_speed(engine, args.generate, args.reps, n, sampling)
        if args.share:
            run |= {"sampling": asdict(sampling)}
            print(json.dumps(bench.shared(engine, run, asdict(chat)), indent=2))
            return
        if args.json:
            print(json.dumps(head | asdict(chat)))
            return
        each = f", {n} at once, {chat.decode / n:.1f} each" if n > 1 else ""
        print(title)
        print(f"  chat tg{args.generate:<4} {chat.decode:8.1f} tok/s   "
              f"{chat.per_step:.2f} tokens a step{each}")  # fmt: skip
        return
    result = bench.speed(engine, args.prompt, args.generate, args.reps, n)
    if args.share:
        print(json.dumps(bench.shared(engine, run, asdict(result)), indent=2))
        return
    if args.json:
        print(json.dumps(head | asdict(result)))
        return
    print(title)
    print(f"  pp{args.prompt:<6} {result.prefill:10.1f} tok/s")
    gbs = f"{result.weight_gbs:.0f} GB/s of weights"
    each = f", {n} at once, {result.decode / n:.1f} each" if n > 1 else ""
    print(f"  tg{args.generate:<6} {result.decode:10.1f} tok/s   {gbs}{each}")


def _perplexity(args: argparse.Namespace) -> None:
    from leatengine import bench
    from leatengine.engine import Engine

    with _said():
        ctx = bench.base_chunk(args.kl_base) if args.kl_base else args.ctx
        # a chunk in one run
        engine = Engine(args.model, max_context=ctx, prefill_chunk=ctx, cache_type=args.cache_type)
        if args.kl_base:
            result = bench.kl_divergence(engine, args.kl_base, args.chunks, args.decode)
        else:
            # its bytes as they are, \r\n too, as llama-perplexity reads them
            text = args.text.read_bytes().decode()
            result = bench.perplexity(engine, text, args.ctx, args.chunks, args.decode)
    if args.json:
        print(json.dumps({"model": model_name(engine.gguf.path)} | asdict(result)))
        return
    print(f"perplexity {result.perplexity:.4f}")
    if result.kl_mean is not None:
        print(f"KL mean {result.kl_mean:.6f}  p99 {result.kl_p99:.6f}  max {result.kl_max:.6f}")
        print(f"same top token {result.top1:.2%}")


def _doctor(args: argparse.Namespace) -> None:
    from leatengine import doctor

    found = doctor.checks()
    print(json.dumps(doctor.as_json(found)) if args.json else doctor.report(found))
    if any(c.status == "fail" for c in found):
        raise SystemExit(1)  # as a script waiting on a ready machine reads it
