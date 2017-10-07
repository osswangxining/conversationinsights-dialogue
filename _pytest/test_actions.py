from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from conversationinsights.events import Restart

from conversationinsights.actions.action import ActionRestart
from conversationinsights.trackers import DialogueStateTracker
import logging

logger = logging.getLogger(__name__)

def test_restart(default_dispatcher, default_domain):
    tracker = DialogueStateTracker("default", default_domain.slots, default_domain.topics, default_domain.default_topic)
    events = ActionRestart().run(default_dispatcher, tracker, default_domain)
    logger.info(events[0])
    logger.info("conversation:{}".format(tracker.conversation()))
    logger.info("slots:{}".format(tracker.slots))
    logger.info("events:{}".format(tracker.events))
    logger.info("default_topic:{}".format(tracker.default_topic.name))
    logger.info("data:{}".format(tracker.data))
    logger.info("topics:{}".format(tracker.topics))
    assert events == [Restart()]
