#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Watches for the appearance of a file."""

import argparse
import asyncio
import os.path


@asyncio.coroutine
def watch_for_file(file_path, interval=1):
    while True:
        if not os.path.exists(file_path):
            print("{} not found yet.".format(file_path))
            yield from asyncio.sleep(interval)
        else:
            print("{} found!".format(file_path))
            break


def make_cli_parser():
    cli_parser = argparse.ArgumentParser(description=__doc__)
    cli_parser.add_argument('file_path')
    return cli_parser


def main(argv=None):
    cli_parser = make_cli_parser()
    args = cli_parser.parse_args(argv)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(watch_for_file(args.file_path))

if __name__ == '__main__':
    main()