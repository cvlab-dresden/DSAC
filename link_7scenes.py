#!/usr/bin/env python3
# link_7scenes.py ---
#
# Filename: link_7scenes.py
# Description: Simple script to setup symlinks to the actual 7scene dataset
# Author: Kwang Moo Yi
# Maintainer:
# Created: Wed Jul 12 11:46:25 2017 (+0200)
# Version:
# Package-Requires: (argparse, os, sys, parse)
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#

# Code:

from __future__ import print_function

import argparse
import os
import sys

from parse import parse


def link_file(src, dest):
    """Links individual files, overwriting if it exists."""
    # remove target symlink if it exists
    if os.path.islink(dest):
        os.unlink(dest)
    # Link
    os.symlink(src, dest)


def link_data(data_dir, dest_dir):
    """Links data from data dir to dest dir """

    # Get absolute path
    abs_data_dir = os.path.abspath(data_dir)
    abs_dest_dir = os.path.abspath(dest_dir)

    print("Linking files...")
    print("data_dir: {}".format(abs_data_dir))
    print("dest_dir: {}".format(abs_dest_dir))

    # For both train and test data
    for set_type in ["Train", "Test"]:
        # Parse the split file to get the list of train and test splits
        split_file = os.path.join(abs_data_dir, set_type + "Split.txt")
        splits = open(split_file, "r").readlines()
        indices = [parse("{}{res:d}", _s)["res"] for _s in splits]
        subdirs = [
            os.path.join(abs_data_dir, "seq-{:02d}".format(_i))
            for _i in indices
        ]
        # Find all files (except the suffix)
        file_names = []
        for _dir in subdirs:
            suffix = ".color.png"
            file_names += [
                os.path.join(_dir, _f.rstrip(suffix))
                for _f in os.listdir(_dir)
                if _f.endswith(suffix)
            ]
        # Make sure nothing is going wrong by looking at the list of file names
        assert len(file_names) > 0
        # Now make symlinks
        print_prefix = "Linking for " + set_type + " ... "
        for i, old_file_name in enumerate(file_names):
            print(
                "\r{} {}/{}".format(print_prefix, i + 1, len(file_names)),
                end="",
            )
            sys.stdout.flush()
            # For some reason the folder naming strategy is incosistent...
            if set_type == "Train":
                suffix = "training"
            else:
                suffix = "test"
            prefix = os.path.join(abs_dest_dir, suffix, "scene")
            # For the color images
            new_file_name = os.path.join(
                prefix, "rgb_noseg",
                "{:09d}.png".format(i))
            if args.dry_run:
                print("{} --> {}".format(
                    old_file_name + ".color.png", new_file_name))
            else:
                link_file(old_file_name + ".color.png", new_file_name)
            # For the depth images
            new_file_name = os.path.join(
                prefix, "depth_noseg",
                "{:09d}.png".format(i))
            if args.dry_run:
                print("{} --> {}".format(
                    old_file_name + ".depth.png", new_file_name))
            else:
                # remove target file if it exists
                if os.path.exists(new_file_name):
                    os.remove(new_file_name)
                link_file(old_file_name + ".depth.png", new_file_name)
            # For the pose files
            new_file_name = os.path.join(
                prefix, "poses",
                "{:09d}.txt".format(i))
            if args.dry_run:
                print("{} --> {}".format(
                    old_file_name + ".pose.txt", new_file_name))
            else:
                link_file(old_file_name + ".pose.txt", new_file_name)
        print("\r{} done.                  ".format(print_prefix))


def main(args, unparsed):

    # if dataset_dir and dest_dir is not defined, or if we have leftover
    # arguments, simply print usage and exit.
    if args.dest_dir == "" or args.data_dir == "":
        print("Usage: link_7scenes.py --data_dir=<data_dir> "
              "--dest_dir=<dest_dir> --dry_run=True/False")
        sys.exit(1)
    else:
        link_data(args.data_dir, args.dest_dir)


if __name__ == '__main__':
    # Create argparse parser
    parser = argparse.ArgumentParser()
    # make bool a type
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # The arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Location of the 7scenes dataset subset. "
        "E.g. chess, fire, etc."
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        default="",
        help="Destination folder to link files to. "
        "E.g. ./7scenes/7scenes_chess"
    )
    parser.add_argument(
        "--dry_run",
        type="bool",
        default=True,
        help="If true, will simply return what the script will link. "
        "By default it is set to True."
    )
    args, unparsed = parser.parse_known_args()
    main(args, unparsed)

#
# link_7scenes.py ends here
