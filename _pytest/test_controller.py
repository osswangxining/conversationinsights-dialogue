from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from conversationinsights.controller import MessageProcessor
from conversationinsights.channels import UserMessage
from conversationinsights.featurizers import BinaryFeaturizer
from conversationinsights.interpreter import RegexInterpreter
from conversationinsights.channels.console import ConsoleOutputChannel
from conversationinsights.policies import PolicyTrainer
from conversationinsights.policies.ensemble import SimplePolicyEnsemble
from conversationinsights.policies.scoring_policy import ScoringPolicy
from conversationinsights.tracker_store import InMemoryTrackerStore


def test_controller(default_domain, capsys):
    story_filename = "data/dsl_stories/stories_defaultdomain.md"
    ensemble = SimplePolicyEnsemble([ScoringPolicy()])
    interpreter = RegexInterpreter()

    PolicyTrainer(ensemble, default_domain, BinaryFeaturizer()).train(
            story_filename,
            max_history=3)

    tracker_store = InMemoryTrackerStore(default_domain)
    processor = MessageProcessor(interpreter, ensemble, default_domain, tracker_store)

    processor.handle_message(UserMessage("_greet", ConsoleOutputChannel()))
    out, _ = capsys.readouterr()
    assert "hey there!" in out
