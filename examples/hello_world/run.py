from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from conversationinsights.actions.action import ACTION_LISTEN_NAME
from conversationinsights.agent import Agent
from conversationinsights.controller import Controller
from conversationinsights.domain import TemplateDomain
from conversationinsights.featurizers import BinaryFeaturizer
from conversationinsights.interpreter import NaturalLanguageInterpreter
from conversationinsights.channels.console import ConsoleInputChannel
from conversationinsights.policies import Policy
from conversationinsights.policies.ensemble import SimplePolicyEnsemble
from conversationinsights.tracker_store import InMemoryTrackerStore
import numpy as np
from conversationinsights.util import one_hot

logger = logging.getLogger(__name__)


class SimplePolicy(Policy):
    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]

        responses = {
            "greet": 3,
        }
        response = None
        if tracker.latest_action_id_str == ACTION_LISTEN_NAME:
            key = tracker.latest_message.intent["name"]
            action = responses[key] if key in responses else 2
            response = one_hot(action, domain.num_actions)
            logger.info(response)
        else:
            response = np.zeros(domain.num_actions)
            logger.info(response)

        return response


class HelloInterpreter(NaturalLanguageInterpreter):
    def parse(self, message):
        intent = "greet" if 'hello' in message else "default"
        return {
            "text": message,
            "intent": {"name": intent, "confidence": 1.0},
            "entities": []
        }


def run_hello_world(serve_forever=True):
    default_domain = TemplateDomain.load("examples/default_domain.yml")
    agent = Agent(
        default_domain,
        policies=[SimplePolicy()],
        interpreter=HelloInterpreter(),
        tracker_store=InMemoryTrackerStore(default_domain))

    if serve_forever:
        # Attach the commandline input to the controller to handle all incomming messages from that channel
        agent.handle_channel(ConsoleInputChannel())

    return agent


if __name__ == '__main__':
    run_hello_world()
