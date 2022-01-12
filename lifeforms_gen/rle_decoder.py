import re
from dataclasses import dataclass

import numpy as np


@dataclass
class RleHeader:
    """
    RLE Run Length Encoded header object, Data object.
    Fields:
        width - width of the pattern
        height - height of the pattern
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height


class RleRawHeader:
    """
    RLE Run Length Encoded raw header object.
    Fields:
        raw_header - raw content from the rle file
    """
    HEADER_PATTERN = re.compile("x = (?P<rows>\\d+),\\s*y = (?P<cols>\\d+)(,\\s*rule = B3.S23)?")

    def __init__(self, raw_header):
        self.raw_header = raw_header

    def parse(self) -> RleHeader:
        pattern = self.HEADER_PATTERN.match(self.raw_header)
        if pattern is None:
            raise ValueError(f"Header is not valid {self.raw_header}")

        return RleHeader(
            width=int(pattern['rows']),
            height=int(pattern['cols']))


class RleRawContent:
    CONTENT_PATTERN = \
        re.compile(
            "(?P<dead_one>b)|(?P<alive_one>o)|(?P<endline>\\$)|(?P<end>!)|(?P<dead_many>\\d+b)|(?P<alive_many>\\d+o)|(?P<many_endline>\\d+\\$)")

    CELLS_LENGTH_INTERPRETER = re.compile("(?P<length>\\d+)[ob$]")

    def __init__(self, raw_content):
        self._pattern = raw_content

    def get_tokens_stream(self):
        current_token = self.CONTENT_PATTERN.match(self._pattern)
        while current_token is not None:
            group_dict = current_token.groupdict()
            for key, value in group_dict.items():
                if value is None:
                    continue
                if key == "end":
                    yield EndToken()
                if key == "endline":
                    yield EndLineToken(1)
                if key == "dead_one":
                    yield DeadToken(length=1)
                if key == "alive_one":
                    yield AliveToken(length=1)
                if key == "dead_many":
                    length = self.CELLS_LENGTH_INTERPRETER.match(value)
                    yield DeadToken(length=int(length.groupdict()["length"]))
                if key == "alive_many":
                    length = self.CELLS_LENGTH_INTERPRETER.match(value)
                    yield AliveToken(length=int(length.groupdict()["length"]))
                if key == "many_endline":
                    length = self.CELLS_LENGTH_INTERPRETER.match(value)
                    yield EndLineToken(length=int(length.groupdict()["length"]))
            self._pattern = self._pattern[current_token.end():]
            current_token = self.CONTENT_PATTERN.match(self._pattern)


def decode(raw_content: str) -> np.ndarray:
    def get_header(lines):
        return next(filter(lambda line: not line.strip().startswith("#"), lines))

    def get_raw_pattern(lines):
        gen = filter(lambda line: not line.strip().startswith("#"), lines)
        _ = next(gen)

        res = ""
        for line in gen:
            res += line.strip()
        return res

    header = RleRawHeader(get_header(raw_content.splitlines())).parse()
    board = np.zeros((header.height, header.width))

    tokens_stream = RleRawContent(get_raw_pattern(raw_content.splitlines())).get_tokens_stream()
    row = 0
    col = 0
    for token in tokens_stream:
        if isinstance(token, EndToken):
            continue
        if isinstance(token, EndLineToken):
            row += token.length
            col = 0
        if isinstance(token, DeadToken):
            col += token.length
        if isinstance(token, AliveToken):
            for _ in range(token.length):
                board[row, col] = 1
                col += 1
    return board


def decode_header(raw_content):
    def get_header(lines):
        return next(filter(lambda line: not line.strip().startswith("#"), lines))

    return RleRawHeader(get_header(raw_content.splitlines())).parse()


class EndToken:
    pass


class EndLineToken:
    def __init__(self, length):
        self.length = length


class AliveToken:
    def __init__(self, length):
        self.length = length


class DeadToken:
    def __init__(self, length):
        self.length = length
