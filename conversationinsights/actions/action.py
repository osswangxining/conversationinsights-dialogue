from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import typing
from typing import List
from typing import Text

if typing.TYPE_CHECKING:
    from conversationinsights.trackers import DialogueStateTracker
    from conversationinsights.dispatcher import Dispatcher
    from conversationinsights.events import Event
    from conversationinsights.domain import Domain

logger = logging.getLogger(__name__)

ACTION_LISTEN_NAME = "action_listen"

ACTION_RESTART_NAME = "action_restart"


class Action(object):
    """Next action to be taken in response to a dialogue state."""

    def resets_topic(self):
        # type: () -> bool
        """Indicator if this action resets the topic when run."""

        return False

    def id_str(self):
        # type: () -> Text
        """Id of this action."""

        return "action_" + self.name()

    def name(self):
        # type: () -> Text
        """Unique identifier of this simple action.

        Key will be prepended with `action_` to form the name and
        with `utter_` to find the utterance."""

        raise NotImplementedError

    def run(self, dispatcher, tracker, domain):
        # type: (Dispatcher, DialogueStateTracker, Domain) -> List[Event]
        """Execute the side effects of this action.

        Return a list of events (i.e. instructions to update tracker state)

        :param tracker: user state tracker
        :param dispatcher: communication channel
        :param domain: bots custom domain
        """

        raise NotImplementedError

    def __str__(self):
        return "Action('{}')".format(self.id_str())


class UtterAction(Action):
    """An action which only effect is to utter a template during its `run` method call.

    Both, name and utter template, need to be specified using the `name` method."""

    def __init__(self, name):
        self._name = name

    def run(self, dispatcher, tracker, domain):
        """Simple run implementation uttering the (hopefully defined) template."""

        dispatcher.utter_template("utter_" + self.name())
        return []

    def name(self):
        return self._name

    def __str__(self):
        return "UtterAction('{}')".format(self.id_str())


class ActionListen(Action):
    """The first action in any turn - bot waits for a user message.

    The bot should stop taking further actions and wait for the user to say something."""

    def name(self):
        return ACTION_LISTEN_NAME[len("action_"):]

    def run(self, dispatcher, tracker, domain):
        return []


class ActionRestart(Action):
    """Resets the tracker to its initial state. Utters the restart template if available."""

    def name(self):
        return ACTION_RESTART_NAME[len("action_"):]

    def run(self, dispatcher, tracker, domain):
        from conversationinsights.trackers import Restart

        # only utter the template if it is available
        if domain.random_template_for("utter_restart") is not None:
            dispatcher.utter_template("utter_restart")
        return [Restart()]
