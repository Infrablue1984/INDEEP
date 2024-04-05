import glob
import os
import pathlib
import re
import sys
import tempfile
from os import path
from shutil import copy


class Config(object):
    # see https://flask.palletsprojects.com/en/1.1.x/config/
    SECRET_KEY = (
        os.environ.get("SECRET_KEY")
        or "slndfsl-wefn2YPL1!24-234opsxOJk__shwQevWfwm4-225523fw-fwe33023"
    )
    # Directory of the program
    BUNDLE_DIR = pathlib.Path(__file__).parent.parent.absolute()
    # get the path of the python csv configuration files
    KEYDATA_DIR = os.path.join(BUNDLE_DIR, "keydata")
    # get the path of source files
    AGENT_MODEL_DIR = None
    # get the path of the scenario
    SCENARIO_FILE_DIR = None
    # get the path of the region/population data
    REGPOP_FILE_DIR = None
    # get the path of the agent files
    AGENT_FILE_DIR = None
    # get the path of the network files
    NETWORK_FILE_DIR = None
    # get the path of the python module or bundle
    LOG_FILE = os.path.join(tempfile.gettempdir(), "bigauge.log")
    # max content size. This is necessary to a avoid huge uploads.
    MAX_CONTENT_LENGTH = 1024 * 1024
    # Size of the diagrams
    DIAGRAM_WIDTH = 1000
    DIAGRAM_HEIGHT = 680

    READ_ONLY_SCENARIO_LIST = []

    CUSTOM_SCENARIO_PREFIX = {"SIR__", "SEEIRD__", "CladeX__", "AGENT__"}

    KEYDATA_SCENARIO_PREFIX = {"SIR_", "SEEIRD_", "CladeX_", "AGENT_"}

    # if "Darwin" in platform.system() :
    abs_path = pathlib.Path(__file__).parent.parent.absolute()
    user_dir = os.path.join(abs_path, "data")

    if not os.path.exists(user_dir):
        os.mkdir(user_dir)

    AGENT_MODEL_DIR = os.path.join("")
    # AGENT_MODEL_DIR = AGENT_MODEL_DIR.replace(" ", r"\ ")  # remove empty spaces

    REGPOP_FILE_DIR = os.path.join(user_dir, "inputs", "social_data")

    if not os.path.exists(REGPOP_FILE_DIR):
        os.mkdir(REGPOP_FILE_DIR)
        src_dir = os.path.join(BUNDLE_DIR, "files", "social_data", "*.xlsx")

        for file in glob.glob(src_dir):
            copy(file, REGPOP_FILE_DIR)

    AGENT_FILE_DIR = os.path.join(user_dir, "inputs", "agent_files")

    if not os.path.exists(AGENT_FILE_DIR):
        os.mkdir(AGENT_FILE_DIR)
        src_dir = os.path.join(BUNDLE_DIR, "files", "social_data", "*.xlsx")

        for file in glob.glob(src_dir):
            copy(file, AGENT_FILE_DIR)

    NETWORK_FILE_DIR = os.path.join(user_dir, "inputs", "network_files")

    if not os.path.exists(NETWORK_FILE_DIR):
        os.mkdir(NETWORK_FILE_DIR)
        src_dir = os.path.join(BUNDLE_DIR, "files", "social_data", "*.xlsx")

        for file in glob.glob(src_dir):
            copy(file, NETWORK_FILE_DIR)

    INFO_FILE_DIR = os.path.join(user_dir, "inputs", "info_file")

    if not os.path.exists(INFO_FILE_DIR):
        os.mkdir(INFO_FILE_DIR)
        src_dir = os.path.join(BUNDLE_DIR, "files", "social_data", "*.xlsx")

        for file in glob.glob(src_dir):
            copy(file, INFO_FILE_DIR)

    OUTPUT_FILE_DIR = os.path.join(user_dir, "outputs")

    SCENARIO_FILE_DIR = os.path.join(user_dir, "inputs", "scenario")

    if not os.path.exists(SCENARIO_FILE_DIR):
        os.mkdir(SCENARIO_FILE_DIR)
        src_dir = os.path.join(BUNDLE_DIR, "files", "social_data", "*.xlsx")

        for file in glob.glob(src_dir):
            copy(file, SCENARIO_FILE_DIR)
