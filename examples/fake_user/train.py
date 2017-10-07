from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys

from examples.fake_user.fake_user import Customer
from conversationinsights.agent import Agent
from conversationinsights.controller import MessageProcessor
from conversationinsights.domain import TemplateDomain
from conversationinsights.featurizers import BinaryFeaturizer
from conversationinsights.interpreter import RegexInterpreter
from conversationinsights.channels import UserMessage
from conversationinsights.channels.console import ConsoleOutputChannel, ConsoleInputChannel
from conversationinsights.policies.keras_policy import KerasPolicy
from conversationinsights.policies.memoization import MemoizationPolicy
from conversationinsights.policies.online_policy_trainer import OnlinePolicyTrainer
from conversationinsights.tracker_store import InMemoryTrackerStore

logger = logging.getLogger(__name__)


def run_fake_user(input_channel, max_training_samples=10, serve_forever=True):
    customer = Customer()
    training_data_file = 'examples/babi/data/babi_task5_fu_fewer_actions.md'

    logger.info("Starting to train policy")

    agent = Agent("examples/restaurant_domain.yml",
                  policies=[MemoizationPolicy(), KerasPolicy()],
                  interpreter=RegexInterpreter())

    agent.train_online(training_data_file,
                       input_channel=input_channel,
                       epochs=1,
                       max_training_samples=max_training_samples)

    while serve_forever:
        tracker = agent.tracker_store.retrieve('default')
        back = customer.respond_to_action(tracker)
        if back == 'reset':
            agent.handle_message("_greet", output_channel=ConsoleOutputChannel())
        else:
            agent.handle_message(back, output_channel=ConsoleOutputChannel())

    return agent


if __name__ == '__main__':
    logging.basicConfig(level="INFO")

    if len(sys.argv) < 2 or sys.argv[1] == 'scratch':
        max_training_samples = 10
    elif sys.argv[1] == 'pretrained':
        max_training_samples = -1
    else:
        raise Exception("Choose from pretrained or training from scratch")

    run_fake_user(ConsoleInputChannel(), max_training_samples)
