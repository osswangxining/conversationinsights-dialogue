from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import uuid
from builtins import str


class Event(object):
    """An event is one of the following:
    - something the user has said to the bot (starts a new turn)
    - the topic has been set
    - the bot has taken an action

    Events are logged by the Tracker's log_event method.
    This updates the list of turns so that the current state
    can be recovered by consuming the list of turns.
    """

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)

    def as_story_string(self):
        raise NotImplementedError


class UserUtterance(Event):
    """
    | **Description:** The user has said something to the bot.
    | **Side effects:** a new ``Turn`` will be created in the ``Tracker``
    """

    def __init__(self, text, intent=None, entities=None, parse_data=None):
        self.text = text
        self.intent = intent if intent else {}
        self.entities = entities if entities else []
        self.parse_data = parse_data if parse_data else {"intent": self.intent, "entities": self.entities, "text": text}

    def __hash__(self):
        return hash((self.text, self.intent, tuple(self.entities)))

    def __eq__(self, other):
        if not isinstance(other, UserUtterance):
            return False
        else:
            return (self.text, self.intent, self.entities, self.parse_data) == \
                   (other.text, other.intent, other.entities, other.parse_data)

    def __str__(self):
        return "Utterance(text: {}, intent: {}, entities: {})".format(self.text, self.intent, self.entities)

    @staticmethod
    def empty():
        return UserUtterance(None)

    def as_story_string(self):
        if self.intent:
            if self.entities:
                es = ','.join(['{}={}'.format(ent['entity'], ent['value']) for ent in self.entities])
                ent_string = "[{}]".format(es)
            else:
                ent_string = ""

            return "_{}{}".format(self.intent.get("name", ""), ent_string)
        else:
            return self.text


class SetTopic(Event):
    """
    | **Description:** The topic of conversation has changed.
    | **Side effects:** self.topic will be pushed on to ``Tracker.topic_stack``
    """

    def __init__(self, topic):
        self.topic = topic

    def __str__(self):
        return "SetTopic(topic: {})".format(self.topic)

    def __hash__(self):
        return hash(self.topic)

    def __eq__(self, other):
        if not isinstance(other, SetTopic):
            return False
        else:
            return self.topic == other.topic

    def as_story_string(self):
        return "topic[{}]".format(self.topic)


class SetSlot(Event):
    """
    | **Description:** The user has specified their preference for the value of a ``slot``.
    | **Side effects:** the ``Tracker``'s slots will be updated so that ``tracker.slots[key]=value``
    """

    def __init__(self, key, value=None):
        self.key = key
        self.value = value

    def __str__(self):
        return "SetSlot(key: {}, value: {})".format(self.key, self.value)

    def __hash__(self):
        return hash((self.key, self.value))

    def __eq__(self, other):
        if not isinstance(other, SetSlot):
            return False
        else:
            return (self.key, self.value) == (other.key, other.value)

    def as_story_string(self):
        return "slot{}".format(json.dumps({self.key: self.value}))


class Restart(Event):
    """
    | **Description:** Conversation should start over & history wiped.
    | **Side effects:** the ``Tracker`` will be reinitialised
    """

    def __hash__(self):
        return hash(32143124312)

    def __eq__(self, other):
        return isinstance(other, Restart)

    def __str__(self):
        return "Restart()"

    def as_story_string(self):
        return "restart"


class PauseConversation(Event):
    """
    | **Description:** Ignore messages from the user to let a human take over.
    | **Side effects:** the ``Tracker``'s ``paused`` attribute will be set to ``True``
    """

    def __hash__(self):
        return hash(32143124313)

    def __eq__(self, other):
        return isinstance(other, PauseConversation)

    def __str__(self):
        return "Pause()"

    def as_story_string(self):
        return "pause"


class ResumeConversation(Event):
    """
    | **Description:** Bot takes over conversation. Inverse of ``PauseConversation``.
    | **Side effects:** the ``Tracker``'s ``paused`` attribute will be set to ``False``
    """

    def __hash__(self):
        return hash(32143124314)

    def __eq__(self, other):
        return isinstance(other, ResumeConversation)

    def __str__(self):
        return "Resume()"

    def as_story_string(self):
        return "resume"


class RevertLastUserUtterance(Event):
    """
    | **Description:** Bot undoes its last action. Shouldn't be used during actual user interactions, mostly for train.
    | **Side effects:** the ``Tracker``'s last turn is removed
    """

    def __hash__(self):
        return hash(32143124315)

    def __eq__(self, other):
        return isinstance(other, RevertLastUserUtterance)

    def as_story_string(self):
        return "revert"


class ResetAllSlots(Event):
    """
    | **Description:** Conversation should start over & history wiped.
    | **Side effects:** the ``Tracker`` will be reinitialised
    """

    def __hash__(self):
        return hash(32143124316)

    def __eq__(self, other):
        return isinstance(other, ResetAllSlots)

    def __str__(self):
        return "ResetAllSlots()"

    def as_story_string(self):
        return "reset_slots"


class Reminder(Event):
    """
    | **Description:** Allows asynchronous scheduling of action execution
    | **Side effects:** the message processor will schedule an action to be run at the trigger date
    """

    def __init__(self, action_name, trigger_date_time, id=None, kill_on_user_message=True):
        """Creates the reminder
        :param action_name: name of the action to be scheduled
        :param trigger_date: date at which the execution of the action should be triggered
        :param id: id of the reminder. if there are multiple reminders with the same id only the last will be run
        :param kill_on_user_message: ``True`` means a user message before the trigger date will abort the reminder
        """

        self.action_name = action_name
        self.trigger_date_time = trigger_date_time
        self.kill_on_user_message = kill_on_user_message
        self.id = id if id is not None else str(uuid.uuid1())

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Reminder):
            return False
        else:
            return self.id == other.id

    def __str__(self):
        return "Reminder(action: {}, trigger_date: {}, id: {})".format(
                self.action_name, self.trigger_date_time, self.id)

    def as_story_string(self):
        return "reminder{}".format(json.dumps({"action": self.action_name,
                                               "date_time": self.trigger_date_time,
                                               "id": self.id,
                                               "kill_on_user_msg": self.kill_on_user_message}))


class ExecutedAction(Event):
    """
    | **Description:** An operation describes an action taken + its result.
     It comprises an action and a list of events.
    | **Side effects:** operations will be appended to the latest ``Turn`` in the ``Tracker.turns``
    """

    def __init__(self, action_name):
        self.action_name = action_name
        self.unpredictable = False

    def __str__(self):
        return "ExecutedAction(action: {})".format(self.action_name)

    def __hash__(self):
        return hash(self.action_name)

    def __eq__(self, other):
        if not isinstance(other, ExecutedAction):
            return False
        else:
            return self.action_name == other.action_name

    def as_story_string(self):
        return self.action_name
