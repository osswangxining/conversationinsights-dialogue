from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import copy
from collections import deque
import io

import jsonpickle
import typing
from typing import Generator
from typing import List
from typing import Optional

from conversationinsights.actions.action import ActionListen
from conversationinsights.conversation import Dialogue
from conversationinsights.events import *
from conversationinsights.util import TopicStack

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if typing.TYPE_CHECKING:
    from conversationinsights.actions import Action


class DialogueStateTracker(object):
    """Maintains the state of a conversation."""

    MAX_EVENT_HISTORY = 500

    def __init__(self, sender_id, slots, topics=None, default_topic=None):
        """Initialize the tracker.

        A set of events can be stored externally, and we will run through all of them to get the current state.
        The tracker will represent all the information we captured while processing messages of the dialogue."""

        # id of the source of the messages
        self.sender_id = sender_id
        # available topics in the domain
        self.topics = topics if topics is not None else []
        # default topic of the domain
        self.default_topic = default_topic
        # slots that can be filled in this domain
        self.slots = {slot.name: copy.deepcopy(slot) for slot in slots}
        self.reset_slots()
        # if tracker is paused, no actions should be taken (e.g. a handover happened, a human will continue texting)
        self.paused = False
        # data we collected (e.g. from external sources)
        self.data = {}
        # A deterministically scheduled action to be executed next
        self.follow_up_action = None
        # list of previously seen events
        self.events = DialogueStateTracker.create_events([])
        # topic tracking
        self.topic_stack = TopicStack(self.topics, [], default_topic)

    def reset(self):
        # type: () -> None
        """Reset tracker to its initial state."""
        self.__init__(self.sender_id, self.slots.values(), self.topics, self.default_topic)

    def reset_slots(self):
        for slot in self.slots.values():
            slot.reset()

    def get_slot(self, key):
        if key in self.slots:
            return self.slots[key].value
        else:
            logger.info("Tried to access non existent slot '{}'".format(key))
            return None

    def set_slot(self, key, value):
        if key in self.slots:
            self.slots[key].value = value
        else:
            logger.warn("Tried to set non existent slot '{}'".format(key))

    @property
    def latest_action_id_str(self):
        for event in reversed(self.events):
            if isinstance(event, ExecutedAction):
                return event.action_name
        return None

    @property
    def latest_action_ids(self):
        return [event.action_name for event in self.events if isinstance(event, ExecutedAction)]

    @property
    def latest_message(self):
        """Information about the previous parsed message."""

        # information about the previous parsed message
        # setting it to empty avoids setting latest message to `None` and the need to check everywhere
        for event in reversed(self.events):
            if isinstance(event, UserUtterance):
                return event
        return UserUtterance.empty()

    def generate_all_prior_states(self):
        # type: () -> Generator[DialogueStateTracker, None, None]
        """Return a generator of the previous states of this tracker, representing its state before each action."""
        from conversationinsights.channels import UserMessage

        tracker = DialogueStateTracker(UserMessage.DEFAULT_SENDER,
                                       self.slots.values(),
                                       self.topics,
                                       self.default_topic)
        logger.debug("slots: {}".format(self.slots))
        logger.debug("topics: {}".format(self.topics))
        logger.debug("default_topic: {}".format(self.default_topic.name if self.default_topic is not None else "None"))
        logger.debug("self.conversation():{}".format(self.conversation()))

        for event in self.events:
            logger.debug("event:{}".format(event.as_story_string()))

            if isinstance(event, ExecutedAction) and not event.unpredictable:
                logger.debug("event.unpredictable:{}".format(event.unpredictable))
                yield tracker
            tracker.log_event(event)

        yield tracker  # yields the final state

    def update_with_dialogue(self, dialogue):
        # type: (Dialogue) -> None
        """
        Use a serialised `Dialogue` (as is persisted in a ``TrackerStore``)
        to update the state of the tracker. If the tracker is blank before calling
        this method, the final state will be identical to the tracker from which
        the dialogue was created.
        """
        if not isinstance(dialogue, Dialogue):
            raise ValueError("story {0} is not of type Dialogue. Have you deserialized it?".format(dialogue))

        self.reset()
        self.append_dialogue(dialogue)

    def append_dialogue(self, dialogue):
        # type: (Dialogue) -> None
        """Appends the events from the dialogue to the current state of the tracker."""

        for event in dialogue.events:
            self.log_event(event)

    @classmethod
    def create_events(cls, events):
        # type: (List[Event]) -> deque

        if events and not isinstance(events[0], Event):  # pragma: no cover
            raise ValueError("events, if given, must be a list of events")
        return deque(events, cls.MAX_EVENT_HISTORY)

    def as_dialogue(self):
        """
        Return a ``Dialogue`` object containing all of the turns in this dialogue.
        This can be serialised and later used to recover the state of this tracker exactly.
        """
        return Dialogue(self.sender_id, list(self.events))

    def log_event(self, event):
        # type: (Event) -> None
        """Modify the state of the tracker according to an ``Event`` which took place in a conversation."""

        if not isinstance(event, Event):  # pragma: no cover
            raise ValueError("event to log must be an instance of a subclass of Event.")

        if isinstance(event, UserUtterance):

            if event.text == '_export':
                self.export_stories_to_file()
            else:
                self.events.append(event)

        elif isinstance(event, ExecutedAction):  # operations can contain nested events
            self.events.append(event)

        elif isinstance(event, SetTopic):
            self.topic_stack.push(event.topic)
            self.events.append(event)

        elif isinstance(event, SetSlot):
            self.set_slot(event.key, event.value)
            self.events.append(event)

        elif isinstance(event, Restart):
            self.reset()
            self.events.append(ExecutedAction(ActionListen().id_str()))

        elif isinstance(event, ResetAllSlots):
            self.reset_slots()
            self.events.append(event)

        elif isinstance(event, PauseConversation):
            self.paused = True
            self.events.append(event)

        elif isinstance(event, Reminder):
            self.events.append(event)

        elif isinstance(event, RevertLastUserUtterance):
            while self.events:
                removed = self.events.pop()  # removes the most recently added event
                if isinstance(removed, UserUtterance):
                    break

        elif isinstance(event, ResumeConversation):
            self.paused = False
            self.events.append(event)

        else:
            raise ValueError("Unknown event: {}".format(event))

    def conversation(self):
        return {"events": list(self.events)}

    @property
    def previous_topic(self):
        for event in reversed(self.events):
            if isinstance(event, SetTopic):
                return event.topic
        return None

    @property
    def topic(self):
        return self.topic_stack.top

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return jsonpickle.encode(self.as_dialogue()) == jsonpickle.encode(other.as_dialogue()) and \
                   self.sender_id == other.sender_id
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def trigger_follow_up_action(self, action):
        # type: (Action) -> None
        """Triggers another action immediately following the execution of the current one."""

        self.follow_up_action = action

    def clear_follow_up_action(self):
        # type: () -> None
        """Clears follow up action when it was executed"""

        self.follow_up_action = None

    def merge_slots(self, entities=None):
        entities = entities if entities else self.latest_message.entities
        new_slots = [SetSlot(e["entity"], e["value"]) for e in entities if e["entity"] in self.slots.keys()]
        return new_slots

    def export_stories(self):
        from conversationinsights.training_utils.dsl import StoryStep, Story

        story_step = StoryStep()
        for event in self.events:
            story_step.add_event(event)
        story = Story([story_step])
        return story.as_story_string(flat=True)

    def export_stories_to_file(self, export_path="debug.md"):
        with io.open(export_path, 'a') as f:
            f.write(self.export_stories())
