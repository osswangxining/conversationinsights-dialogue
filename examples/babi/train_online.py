from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from examples.babi.run import nlu_model_path
from examples.restaurant_example import RestaurantPolicy
from conversationinsights.agent import Agent
from conversationinsights.channels.file import FileInputChannel
from conversationinsights.domain import TemplateDomain
from conversationinsights.featurizers import BinaryFeaturizer
from conversationinsights.interpreter import RegexInterpreter
from conversationinsights.interpreter import MyNLUHttpInterpreter
from conversationinsights.policies.memoization import MemoizationPolicy

logger = logging.getLogger(__name__)


def run_babi_online():
    training_data_file = 'examples/babi/data/babi_task5_dev_even_smaller.md'
    logger.info("Starting to train policy")
    agent = Agent("examples/restaurant_domain.yml",
                  policies=[MemoizationPolicy(), RestaurantPolicy()],
                  interpreter=RegexInterpreter())

    agent.train_online(training_data_file,
                       input_channel=FileInputChannel(training_data_file,
                                                      message_line_pattern='^\s*\*\s(.*)$',
                                                      max_messages=10),
                       epochs=10)

    agent.interpreter = MyNLUHttpInterpreter("default",
                                                          "token", "http://127.0.0.1:5000")
    return agent


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    run_babi_online()
