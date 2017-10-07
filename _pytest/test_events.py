from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from copy import deepcopy

import pytest as pytest

from conversationinsights.events import UserUtterance, SetTopic, SetSlot, Restart, PauseConversation, ResumeConversation, \
    ExecutedAction, ResetAllSlots, Reminder


@pytest.mark.parametrize("one_event,another_event", [
    (UserUtterance("_greet", "greet", []), UserUtterance("_goodbye", "goodbye", [])),
    (SetTopic("my_topic"), SetTopic("my_other_topic")),
    (SetSlot("my_slot", "value"), SetSlot("my__other_slot", "value")),
    (Restart(), None),
    (PauseConversation(), None),
    (ResumeConversation(), None),
    (ResetAllSlots(), None),
    (ExecutedAction("my_action"), ExecutedAction("my_other_action")),
    (Reminder("my_action", "now"), Reminder("my_other_action", "now")),
])
def test_event_has_proper_implementation(one_event, another_event):
    # equals tests
    assert one_event != another_event, "Same events with different values need to be different"
    assert one_event == deepcopy(one_event), "Event copies need to be the same"
    assert one_event != 42, "Events aren't equal to 42!"

    # hash test
    assert hash(one_event) == hash(deepcopy(one_event)), "Same events should have the same hash"
    assert hash(one_event) != hash(another_event), "Different events should have different hashes"

    # str test
    assert "object at 0x" not in str(one_event), "Event has a proper str method"
