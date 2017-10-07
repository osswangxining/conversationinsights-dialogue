from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import conversationinsights.version

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = conversationinsights.version.__version__
