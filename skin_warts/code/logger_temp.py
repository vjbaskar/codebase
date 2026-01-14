#!/usr/bin/env python

"""
Logging configuration for the project.

This module sets up the logging format and level for the analysis.
It uses the standard logging library to create a logger that outputs formatted
log messages to the console. It is recommended to use this module at the
beginning of your scripts to ensure consistent logging across the analysis.
"""

import logging, os, sys

FORMATS = {
    logging.DEBUG:    "\033[1;38m", # grey
    logging.INFO:     "\033[0;32m", # green
    logging.WARNING:  "\033[1;33m", # yellow
    logging.ERROR:    "\033[0;31m", # red
    logging.CRITICAL: "\033[1;31m", # red bold
}
logger_format = "[%(asctime)s] %(name)s %(levelname)-2s [%(filename)s:%(funcName)s:%(lineno)s] %(message)s"

# Set up logging configuration
logging.basicConfig(
    format=logger_format,
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout, # send to output (no red background)
)
for level, format in FORMATS.items():
    logging.addLevelName(level, format + logging.getLevelName(level) + "\033[0m")

# Name logger with the project name
project_path = os.popen("git rev-parse --show-toplevel 2>&1").read().rstrip()
logger = logging.getLogger(os.path.basename(project_path))

# Set up logging levels
logger.setLevel(logging.INFO)

info = logger.info
warning = logger.warning
debug = logger.debug
error = logger.error
critical = logger.critical
