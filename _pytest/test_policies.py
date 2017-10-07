from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pytest

from conversationinsights.channels import UserMessage
from conversationinsights.domain import TemplateDomain
from conversationinsights.events import UserUtterance
from conversationinsights.featurizers import BinaryFeaturizer
from conversationinsights.policies.keras_policy import KerasPolicy
from conversationinsights.policies.memoization import MemoizationPolicy
from conversationinsights.policies.scoring_policy import ScoringPolicy

# We are going to use class style testing here since unfortunately pytest doesn't support using
# fixtures as arguments to its own parameterize yet (hence, we can't train a policy, declare it as a fixture
# and use the different fixtures of the different policies for the functional tests). Therefore, we are going to
# reverse this and train the policy within a class and collect the tests in a base class.
from conversationinsights.trackers import DialogueStateTracker
from conversationinsights.training_utils import create_stories_from_file
import logging
logger = logging.getLogger(__name__)

def train_data(max_history, domain):
    featurizer = BinaryFeaturizer()
    stories = create_stories_from_file("data/dsl_stories/stories_defaultdomain.md")
    return domain.training_data_from_stories(featurizer, stories, max_history, should_remove_duplicates=True)


class PolicyTestCollection(object):
    """Tests every policy needs to fulfill. Each policy can declare further tests on its own."""

    max_history = 3         # this is the amount of history we test on

    def create_policy(self):
        raise NotImplementedError

    @pytest.fixture(scope="module")
    def trained_policy(self):
        default_domain = TemplateDomain.load("examples/default_domain.yml")
        policy = self.create_policy()
        X, y = train_data(self.max_history, default_domain)
        policy.max_history = self.max_history
        policy.featurizer = BinaryFeaturizer()
        policy.train(X, y, default_domain)
        return policy

    def test_persist_and_load(self, trained_policy, default_domain, tmpdir):
        trained_policy.persist(tmpdir.strpath)
        loaded = trained_policy.__class__.load(tmpdir.strpath, trained_policy.featurizer, trained_policy.max_history)
        stories = create_stories_from_file("data/dsl_stories/stories_defaultdomain.md")

        for story in stories:
            tracker = DialogueStateTracker("default", default_domain.slots)
            tracker.update_with_dialogue(story.as_dialogue("default", default_domain))
            prob1 = loaded.predict_action_probabilities(tracker, default_domain)
            prob2 = trained_policy.predict_action_probabilities(tracker, default_domain)
            logger.info("prob1:{}".format(prob1))
            logger.info("prob2:{}".format(prob2))
            assert prob1 == prob2


    def test_prediction_on_empty_tracker(self, trained_policy, default_domain):
        tracker = DialogueStateTracker(UserMessage.DEFAULT_SENDER,
                                       default_domain.slots,
                                       default_domain.topics,
                                       default_domain.default_topic)
        probabilities = trained_policy.predict_action_probabilities(tracker, default_domain)
        assert len(probabilities) == default_domain.num_actions
        assert max(probabilities) <= 1.0
        assert min(probabilities) >= 0.0

    def test_persist_and_load_empty_policy(self, tmpdir):
        empty_policy = self.create_policy()
        empty_policy.persist(tmpdir.strpath)
        loaded = empty_policy.__class__.load(tmpdir.strpath, BinaryFeaturizer(), empty_policy.max_history)
        assert loaded is not None


class TestKerasPolicy(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self):
        p = KerasPolicy()
        return p


class TestScoringPolicy(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self):
        p = ScoringPolicy()
        return p


class TestMemoizationPolicy(PolicyTestCollection):
    @pytest.fixture(scope="module")
    def create_policy(self):
        p = MemoizationPolicy()
        return p

    def test_memorise(self, trained_policy, default_domain):
        X, y = train_data(self.max_history, default_domain)
        trained_policy.train(X, y, default_domain)

        for ii in range(X.shape[0]):
            assert trained_policy.recall(X[ii, :, :], default_domain) == y[ii]

        assert trained_policy.recall(np.random.randn(default_domain.num_features), default_domain) is None
