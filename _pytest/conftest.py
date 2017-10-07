from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import pytest

from conversationinsights.channels.console import ConsoleOutputChannel
from conversationinsights.dispatcher import Dispatcher
from conversationinsights.domain import TemplateDomain

logging.basicConfig(level="DEBUG")


@pytest.fixture(scope="function")
def default_domain():
    return TemplateDomain.load("examples/default_domain.yml")


@pytest.fixture
def default_dispatcher(default_domain):
    bot = ConsoleOutputChannel()
    return Dispatcher("my-sender", bot, default_domain)
