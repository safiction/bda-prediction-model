"""Read only the two allowed sources. Raw files are never changed."""

import codecs
import csv
import hashlib
from datetime import date, datetime
from pathlib import Path


SOURCES = ("cdr", "tickets")
REQUIRED_COLUMNS = {
    "cdr": {"record_id", "subscriber_key", "event_ts", "ingest_ts",
            "duration_sec", "bytes_up", "bytes_down", "charge"},
    "tickets": {"ticket_id", "subscriber_key", "created_ts", "closed_ts"},
}


def discover_files(data_dir):
    files = []
    root = Path(data_dir).resolve()
    for folder in ("raw", "late_delivery"):
        for source in SOURCES:
            for path in sorted((root / folder / source).glob(f"dt=*/{source}.csv")):
                if not path.resolve().is_relative_to(root / folder / source):
                    raise ValueError(f"Source file escapes allowed directory: {path}")
                partition = date.fromisoformat(path.parent.name.removeprefix("dt="))
                files.append((partition, source, path))
    return sorted(files)


def inspect_file(path, encodings):
    # Проверяем весь файл: дефект кодировки может находиться после первых строк.
    digest = hashlib.sha256()
    decoders = {enc: codecs.getincrementaldecoder(enc)(errors="strict") for enc in encodings}
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            for encoding, decoder in list(decoders.items()):
                try:
                    decoder.decode(chunk)
                except UnicodeDecodeError:
                    del decoders[encoding]
    for encoding, decoder in list(decoders.items()):
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            del decoders[encoding]
    if not decoders:
        raise ValueError(f"No supported encoding: {path}")
    return next(iter(decoders)), digest.hexdigest()


def read_rows(path, encoding, source):
    with path.open(encoding=encoding, newline="") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED_COLUMNS[source] <= set(reader.fieldnames or []):
            raise ValueError(f"Required {source} columns missing in {path}")
        for row in reader:
            yield row


def parse_timestamp(value):
    try:
        text = value.strip()
        # В CDR встречаются ISO-даты и формат ДД.ММ.ГГГГ.
        if len(text) > 10 and text[2] == "." and text[5] == ".":
            parsed = datetime.strptime(text, "%d.%m.%Y %H:%M:%S")
        else:
            parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is not None:
            raise ValueError("Timezone-aware timestamp requires an explicit timezone policy")
        return parsed
    except (ValueError, AttributeError):
        return None
