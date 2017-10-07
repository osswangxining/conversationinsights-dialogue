from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob

import pytest

from examples.hello_world.run import SimplePolicy
from conversationinsights.actions.action import ActionListen
from conversationinsights.agent import Agent
from conversationinsights.conversation import Topic
from conversationinsights.domain import TemplateDomain
from conversationinsights.events import UserUtterance, SetTopic, ExecutedAction
from conversationinsights.featurizers import BinaryFeaturizer
from conversationinsights.interpreter import NaturalLanguageInterpreter
from conversationinsights.tracker_store import InMemoryTrackerStore, RedisTrackerStore
from conversationinsights.trackers import DialogueStateTracker
from conversationinsights.training_utils import create_stories_from_file
from utilities import tracker_from_dialogue_file, read_dialogue_file

import logging
logger = logging.getLogger(__name__)

domain = TemplateDomain.load("data/test_domains/default_with_topic.yml")


def stores_to_be_tested():
    return [RedisTrackerStore(domain, mock=True),
            InMemoryTrackerStore(domain)]


def stores_to_be_tested_ids():
    return ["redis-tracker",
            "in-memory-tracker"]


def test_tracker_duplicate():
    filename = "data/test_dialogues/inform_no_change.json"
    dialogue = read_dialogue_file(filename)
    logger.info("dialogue.events:{}".format(dialogue.events))
    dialogue_topics = set([Topic(t.topic) for t in dialogue.events if isinstance(t, SetTopic)])
    logger.info("dialogue_topics:{}".format(dialogue_topics))
    domain.topics.extend(dialogue_topics)
    logger.info("domain.topics:{}".format(domain))
    tracker = DialogueStateTracker(dialogue.name, domain.slots, domain.topics, domain.default_topic)
    tracker.update_with_dialogue(dialogue)
    num_actions = len([event for event in dialogue.events if isinstance(event, ExecutedAction)])
    logger.info("num_actions:{}".format(num_actions))
    # There is always one duplicated tracker more than we have actions, as the tracker also gets duplicated for the
    # action that would be next (but isn't part of the operations)
    prior_status = list(tracker.generate_all_prior_states())
    logger.info(prior_status[0].conversation())
    assert len(prior_status) == num_actions + 1


@pytest.mark.parametrize("store", stores_to_be_tested(), ids=stores_to_be_tested_ids())
def test_tracker_store_storage_and_retrieval(store):
    tracker = store.get_or_create_tracker("some-id")
    # the retreived tracker should be empty
    assert tracker.sender_id == "some-id"
    assert list(tracker.events) == [ExecutedAction(ActionListen().id_str())]     # Action listen should be in there

    tracker.log_event(UserUtterance("_greet", {"name": "greet", "confidence": 1.0}, []))  # lets log a test message
    assert tracker.latest_message.intent.get("name") == "greet"
    store.save(tracker)

    # retrieving the same tracker should result in the same tracker
    retrieved_tracker = store.get_or_create_tracker("some-id")
    assert retrieved_tracker.sender_id == "some-id"
    assert len(retrieved_tracker.events) == 2
    assert retrieved_tracker.latest_message.intent.get("name") == "greet"

    # getting another tracker should result in an empty tracker again
    other_tracker = store.get_or_create_tracker("some-other-id")
    assert other_tracker.sender_id == "some-other-id"
    assert len(other_tracker.events) == 1


@pytest.mark.parametrize("store", stores_to_be_tested(), ids=stores_to_be_tested_ids())
@pytest.mark.parametrize("filename", glob.glob('data/test_dialogues/*json'))
def test_tracker_store(filename, store):
    tracker = tracker_from_dialogue_file(filename, domain)
    store.save(tracker)
    restored = store.retrieve(tracker.sender_id)
    assert restored == tracker


def test_tracker_write_to_story(tmpdir):
    tracker = tracker_from_dialogue_file("data/test_dialogues/inform_no_change.json", domain)
    p = tmpdir.join("export.md")
    tracker.export_stories_to_file(p.strpath)
    stories = create_stories_from_file(p.strpath, augmentation_factor=0)
    assert len(stories) == 1
    assert len(stories[0].story_steps) == 1
    assert len(stories[0].story_steps[0].events) == 5
    assert stories[0].story_steps[0].start_checkpoints[0] == "STORY_START"
    assert stories[0].story_steps[0].events[3].text == "_deny"


def test_tracker_state_regression(default_domain):
    from conversationinsights.channels import UserMessage

    class HelloInterpreter(NaturalLanguageInterpreter):
        def parse(self, text):
            intent = "greet" if 'hello' in text else "default"
            return {
                "text": text,
                "intent": {"name": intent},
                "entities": []
            }

    agent = Agent(domain, [SimplePolicy()], BinaryFeaturizer(), interpreter=HelloInterpreter())

    n_actions = []
    for i in range(0, 2):
        agent.handle_message("hello")
        tracker = agent.tracker_store.get_or_create_tracker('default')
        n_actions.append(len(tracker.latest_action_ids))
    # Ensures that the tracker has changed between the utterances (and wasn't reset in between them)
    assert n_actions[0] != n_actions[1]
