import unittest

import numpy
import numpy as np

from lifeforms_gen import rle_decoder as decoder


class TestRleDecoder(unittest.TestCase):

    def test_valid_header_with_rule(self):
        raw_content = decoder.RleRawHeader("x = 10, y = 9, rule = B3/S23")
        rle_header = raw_content.parse()
        self.assertEqual(10, rle_header.width)
        self.assertEqual(9, rle_header.height)

    def test_valid_header_without_rule(self):
        raw_content = decoder.RleRawHeader("x = 10, y = 9")
        rle_header = raw_content.parse()
        self.assertEqual(10, rle_header.width)
        self.assertEqual(9, rle_header.height)

    def test_invalid_header(self):
        raw_content = decoder.RleRawHeader("x = 10,")
        with self.assertRaises(ValueError):
            raw_content.parse()

    def test_stream_token_single_values(self):
        stream = decoder.RleRawContent("2b!").get_tokens_stream()

        token = next(stream)
        self.assertIsInstance(token, decoder.DeadToken)
        self.assertEqual(2, token.length)
        token = next(stream)
        self.assertIsInstance(token, decoder.EndToken)

        stream = decoder.RleRawContent("b!").get_tokens_stream()
        token = next(stream)
        self.assertIsInstance(token, decoder.DeadToken)
        self.assertEqual(1, token.length)
        token = next(stream)
        self.assertIsInstance(token, decoder.EndToken)

    def test_decoder(self):
        expected = np.ones((1, 3))
        with open("../data-sources/rle/blinker.rle") as f:
            raw_content = f.read()
            board = decoder.decode(raw_content)
            self.assertTrue(numpy.array_equal(expected, board))

        expected = np.zeros((5, 25))
        # i = 0
        expected[0, 0:2] = 1
        expected[0, 4] = 1
        expected[0, 6:8] = 1
        # i = 1
        expected[1, 0:5] = 1
        expected[1, 6] = 1
        expected[1, 8] = 1
        # i = 2
        expected[2, 8] = 1
        expected[2, 10:13] = 1
        expected[2, 14:17] = 1
        expected[2, 18:21] = 1
        expected[2, 22:25] = 1
        # i = 3
        expected[3, 0:5] = 1
        expected[3, 6] = 1
        expected[3, 8] = 1
        # i = 4
        expected[4, 0:2] = 1
        expected[4, 4] = 1
        expected[4, 6:8] = 1
        with open("../data-sources/rle/blinkerfuse.rle") as f:
            raw_content = f.read()
            board = decoder.decode(raw_content)
            self.assertTrue(np.array_equal(expected, board))