from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import six
from conversationinsights.agent import Agent
from conversationinsights.interpreter import MyNLUHttpInterpreter
from conversationinsights.channels.console import ConsoleInputChannel

if six.PY2:
    nlu_model_path = 'examples/babi/models/nlu/current_py2'
else:
    nlu_model_path = 'examples/babi/models/nlu/current_py3'


def run_babi(serve_forever=True):
    agent = Agent.load("examples/babi/models/policy/current",
                       interpreter=MyNLUHttpInterpreter("default",
                                                          "token", "http://127.0.0.1:5000"))
    if serve_forever:
        agent.handle_channel(ConsoleInputChannel())
    return agent


if __name__ == '__main__':
    logging.basicConfig(level="DEBUG")
    run_babi()
