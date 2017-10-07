from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from builtins import str
from typing import Text, List, Optional, Callable, Any, Union

from conversationinsights.channels import UserMessage, InputChannel, OutputChannel
from conversationinsights.controller import Controller
from conversationinsights.domain import TemplateDomain, Domain
from conversationinsights.featurizers import Featurizer, BinaryFeaturizer
from conversationinsights.interpreter import NaturalLanguageInterpreter
from conversationinsights.policies import PolicyTrainer
from conversationinsights.policies.ensemble import SimplePolicyEnsemble, PolicyEnsemble
from conversationinsights.policies.memoization import MemoizationPolicy
from conversationinsights.policies.online_policy_trainer import OnlinePolicyTrainer
from conversationinsights.tracker_store import InMemoryTrackerStore, TrackerStore


class Agent(object):
    """Interface for common things to do (e.g. train an assistant, or handle messages with an assistant)."""

    def __init__(self, domain, policies=None, featurizer=None, interpreter=None, tracker_store=None):
        self.domain = self._create_domain(domain)
        self.featurizer = self._create_featurizer(featurizer)
        self.policy_ensemble = self._create_ensemble(policies)
        self.tracker_store = self._create_tracker_store(tracker_store, self.domain)
        self.interpreter = NaturalLanguageInterpreter.create(interpreter)

    @classmethod
    def load(cls, path, interpreter=None, tracker_store=None):
        # type: (Text, Union[NaturalLanguageInterpreter, Text, None], Optional[TrackerStore]) -> Agent
        domain = TemplateDomain.load(os.path.join(path, "domain.yml"))
        domain.compare_with_specification(path)  # ensures the domain hasn't changed between test and train
        featurizer = Featurizer.load(path)
        ensemble = PolicyEnsemble.load(path, featurizer)
        _interpreter = NaturalLanguageInterpreter.create(interpreter)
        _tracker_store = cls._create_tracker_store(tracker_store, domain)
        return Agent(domain, ensemble, featurizer, _interpreter, _tracker_store)

    def handle_asynchronous(self, input_channel,
                            message_queue=None,
                            num_processing_threads=1,
                            message_preprocessor=None):
        # type: (InputChannel, Dequeue, int, Optional[Callable[[Text], Text]]) -> None
        """Handle the messages coming from the input channel asynchronously in child threads.

        Spawns a number of threads to handle the messages that reach the input channel."""

        controller = self._create_controller(message_preprocessor)
        controller.handle_asynchronous(input_channel, message_queue, num_processing_threads)

    def handle_message(self, text_message, message_preprocessor=None, output_channel=None):
        # type: (Text, Optional[Callable[[Text], Text]], Optional[OutputChannel]) -> Optional[List[Text]]
        """Handle a single message.

        If a message preprocessor is passed, the message will be passed to that function first and
        the return value is then used as the input for the dialogue engine.

        The return value of this function depends on the ``output_channel`. If the output channel is
        not set (`None`) or set to `CollectingOutputChannel` this function will return
        the messages the bot wants to respond.
        """

        controller = self._create_controller(message_preprocessor)
        return controller.handle_message(UserMessage(text_message, output_channel))

    def handle_channel(self, input_channel, message_preprocessor=None):
        # type: (InputChannel, Optional[Callable[[Text], Text]]) -> None
        """Handle messages coming from the channel."""

        controller = self._create_controller(message_preprocessor)
        controller.handle_channel(input_channel)

    def toggle_memoization(self, activate):
        # type: (bool) -> None
        """If a memoization policy is present in the ensemble, this will toggle the prediction of that policy.

        When set to `false` the Memoization policies present in the policy ensemble will not make any predictions.
        Hence, the prediction result from the ensemble always needs to come from a different
        policy (e.g. `KerasPolicy`). Useful to test prediction capabilities of an ensemble when ignoring memorized
        turns from the training data."""

        for p in self.policy_ensemble.policies:
            if type(p) == MemoizationPolicy:  # explicitly ignore inheritance (e.g. scoring policy)
                p.toggle(activate)

    def train(self, filename=None, **kwargs):
        # type: (Optional[Text], **Any) -> None
        """Train the set policies / policy ensemble using the dialogue data found in the passed filename."""

        trainer = PolicyTrainer(self.policy_ensemble, self.domain, self.featurizer)
        trainer.train(filename, **kwargs)

    def train_online(self, filename=None, input_channel=None, **kwargs):
        # type: (Optional[Text], **Any) -> None
        """Runs an online training session on the set policies / policy ensemble.

        The policies will be pretrained using the data from `filename`. After that the model will get
        trained on dialogues from the input channel. During the dialogue the annotations and state
        of the agent can be changed to correct wrong behaviour."""

        if not self.interpreter:
            raise ValueError("When using online learning, you need to specify an interpreter for the agent to use.")
        trainer = OnlinePolicyTrainer(self.policy_ensemble, self.domain, self.featurizer)
        trainer.train(filename, self.interpreter, input_channel, **kwargs)

    def persist(self, model_path):
        # type: (Text) -> None
        """Persists this agent into a directory for later loading and usage."""

        self.policy_ensemble.persist(model_path)
        self.domain.persist(os.path.join(model_path, "domain.yml"))
        self.domain.persist_specification(model_path)
        self.featurizer.persist(model_path)

    def _ensure_agent_is_prepared(self):
        # type: () -> None
        """Checks that an interpreter and a tracker store are set.

        Necessary before a controller can be instantiated from this agent. Raises an exception if any
        argument is missing."""

        if self.interpreter is None or self.tracker_store is None:
            raise Exception("Agent needs to be prepared before usage. " +
                            "You need to set an interpreter as well as a tracker store.")

    def _create_controller(self, message_preprocessor=None):
        # type: (Callable[[Text], Text]) -> Controller
        """Instantiates a controller based on the set state of the agent."""

        self._ensure_agent_is_prepared()
        return Controller(self.interpreter, self.policy_ensemble, self.domain,
                          self.tracker_store, message_preprocessor)

    @classmethod
    def _create_featurizer(cls, featurizer):
        return featurizer if featurizer is not None else BinaryFeaturizer()

    @classmethod
    def _create_domain(cls, domain):
        if isinstance(domain, str):
            return TemplateDomain.load(domain)
        if isinstance(domain, Domain):
            return domain
        raise ValueError("Invalid param `domain`. Expected a path to a domain specification or a domain instance. " +
                         "But got type '{}' with value '{}'".format(type(domain), domain))

    @classmethod
    def _create_tracker_store(cls, store, domain):
        return store if store is not None else InMemoryTrackerStore(domain)

    @classmethod
    def _create_interpreter(cls, interp):
        return NaturalLanguageInterpreter.create(interp)

    @classmethod
    def _create_ensemble(cls, policies):
        if policies is None:
            return SimplePolicyEnsemble([MemoizationPolicy])
        if isinstance(policies, list):
            return SimplePolicyEnsemble(policies)
        elif isinstance(policies, PolicyEnsemble):
            return policies
        raise ValueError("Invalid param `policies`. Passed object is of type '{}', ".format(type(policies).__name__) +
                         "but should be policy, an array of policies, or a policy ensemble")
