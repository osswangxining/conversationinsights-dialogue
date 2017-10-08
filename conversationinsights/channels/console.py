# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
from builtins import input
from typing import Text

from conversationinsights.channels.channel import UserMessage
from conversationinsights.channels.channel import InputChannel, OutputChannel
from conversationinsights.util import bcolors, print_color


class ConsoleOutputChannel(OutputChannel):
    """Simple bot that outputs the bots messages to the command line."""

    default_output_color = bcolors.OKBLUE

    def send_text_message(self, recipient_id, message):
        # type: (Text, Text) -> None
        print_color(message, self.default_output_color)


class ConsoleInputChannel(InputChannel):
    """Input channel that reads the user messages from the command line."""

    def _record_messages(self, on_message, max_message_limit=None):
        print_color("欢迎使用Conversation Insights智能对话服务，输入消息后回车。 ", bcolors.OKBLUE)
        num_messages = 0
        while max_message_limit is None or num_messages < max_message_limit:
            text = input().strip()
            if six.PY2:
                text = text.decode("utf-8")  # in python 2 input doesn't return unicode values

            if text == '_stop':
                import os
                # sys.exit(1)
                os._exit(1)
            on_message(UserMessage(text, ConsoleOutputChannel()))
            num_messages += 1

    def start_async_listening(self, message_queue):
        self._record_messages(message_queue.enqueue)

    def start_sync_listening(self, message_handler):
        self._record_messages(message_handler)
