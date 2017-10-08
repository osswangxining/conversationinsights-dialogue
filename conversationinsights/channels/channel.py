from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from types import LambdaType

from typing import Text, List, Dict, Any, Optional


class UserMessage(object):
    """Represents an incoming message as well as the channel the responses should be sent to."""

    DEFAULT_SENDER = "default"

    def __init__(self, text, output_channel=None, sender_id=DEFAULT_SENDER):
        # type: (Text, Optional[OutputChannel], Text) -> None
        from conversationinsights.channels.direct import CollectingOutputChannel

        self.output_channel = output_channel if output_channel is not None else CollectingOutputChannel()
        self.text = text
        self.sender_id = sender_id


class InputChannel(object):
    """Input channel base class. Collects messages from some source and puts them into the message queue."""

    def start_async_listening(self, message_queue):
        # type: (Dequeue) -> None
        """Should start to push the incoming messages from this channel into the queue."""
        raise Exception("Input channel doesn't support async listening.")

    def start_sync_listening(self, message_handler):
        # type: (LambdaType) -> None
        """Should call the message handler for every incoming message."""
        raise Exception("Input channel doesn't support sync listening.")


class OutputChannel(object):
    """Output channel base class. Provides sane implementation of the send methods for text only output channels."""

    def send_text_message(self, recipient_id, message):
        # type: (Text, Text) -> None
        """Send a message through this channel."""
        raise NotImplementedError("Output channel needs to implement a send message for simple texts.")

    def send_image_url(self, recipient_id, image_url):
        # type: (Text, Text) -> None
        """Sends an image. Default implementation will just post the url as a string."""
        self.send_text_message(recipient_id, "Image: {}".format(image_url))

    def send_text_with_buttons(self, recipient_id, message, buttons, **kwargs):
        # type: (Text, Text, List[Dict[Text, Any]], **Any) -> None
        """Sends buttons to the output. Default implementation will just post the buttons as a string."""

        self.send_text_message(recipient_id, message)
        for idx, button in enumerate(buttons):
            button_msg = "{0}: {1} ({2})".format(idx + 1, button['title'], button['payload'])
            self.send_text_message(recipient_id, button_msg)

    def send_custom_message(self, recipient_id, elements):
        # type: (Text, List[Dict[Text, Any]]) -> None
        """Sends elements to the output. Default implementation will just post the elements as a string."""

        for element in elements:
            element_msg = "{0} : {1}".format(element['title'], element['subtitle'])
            self.send_text_with_buttons(recipient_id, element_msg, element['buttons'])
