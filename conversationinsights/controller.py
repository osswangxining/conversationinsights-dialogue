from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import sys
from threading import Thread
from types import LambdaType

from apscheduler.schedulers.background import BackgroundScheduler
from typing import Optional, List
from typing import Text

from conversationinsights.actions.action import ActionListen, ActionRestart
from conversationinsights.channels import UserMessage
from conversationinsights.channels.direct import CollectingOutputChannel
from conversationinsights.dispatcher import Dispatcher
from conversationinsights.domain import Domain
from conversationinsights.events import Restart, Reminder
from conversationinsights.events import UserUtterance, ExecutedAction
from conversationinsights.interpreter import NaturalLanguageInterpreter
from conversationinsights.interpreter import RegexInterpreter
from conversationinsights.policies.ensemble import PolicyEnsemble
from conversationinsights.tracker_store import TrackerStore
from conversationinsights.trackers import DialogueStateTracker

scheduler = BackgroundScheduler()
scheduler.start()

try:
    # noinspection PyCompatibility
    from Queue import Queue, Empty
except ImportError:
    # noinspection PyCompatibility
    from queue import Queue, Empty

logger = logging.getLogger(__name__)


class MessageQueue(object):
    def enqueue(self, message):
        # type: (UserMessage) -> None
        """Add a message to the queue."""
        raise NotImplementedError

    def dequeue(self):
        # type: () -> Optional[UserMessage]
        """Remove a message from the queue."""
        raise NotImplementedError


class InMemoryMessageQueue(MessageQueue):
    def __init__(self):
        self.queue = Queue()

    def enqueue(self, message):
        # type: (UserMessage) -> None
        """Add a message to the queue to be handled."""
        self.queue.put(message)

    def dequeue(self):
        # type: () -> Optional[UserMessage]
        """Remove a message from the queue (the one who removes it should also handle it!)"""

        try:
            return self.queue.get(block=True)
        except Empty:
            return None

    def join(self):
        # type: () -> None
        """Wait until all messages in the queue have been processed."""
        self.queue.join()


class Controller(object):
    def __init__(self, interpreter, policy_ensemble, domain, tracker_store, message_preprocessor=None):
        # type: (NaturalLanguageInterpreter, PolicyEnsemble, Domain, TrackerStore, Optional[LambdaType]) -> None
        self.tracker_store = tracker_store
        self.domain = domain
        self.policy_ensemble = policy_ensemble
        self.interpreter = interpreter
        self.threads = []
        self.message_preprocessor = message_preprocessor

    def handle_asynchronous(self, input_channel=None, message_queue=None, num_processing_threads=1):
        # type: (InputChannel, Dequeue, int) -> None
        """Handle the messages coming from the input channel asynchronously in child threads.

        Spawns a number of threads to handle the messages that reach the input channel."""
        if message_queue is None:
            message_queue = InMemoryMessageQueue()
        # hook up input channel
        if input_channel is not None:
            listener_thread = Thread(target=input_channel.start_async_listening, args=[message_queue])
            listener_thread.daemon = True
            listener_thread.start()
            self.threads.append(listener_thread)

        # create message processors
        for i in range(0, num_processing_threads):
            message_processor = self.create_processor()
            processor_thread = Thread(target=message_processor.handle_channel_asynchronous, args=[message_queue])
            processor_thread.daemon = True
            processor_thread.start()
            self.threads.append(processor_thread)

    def handle_channel(self, input_channel=None):
        # type: (InputChannel) -> None
        """Handle messages coming from the channel."""

        message_processor = self.create_processor()
        message_processor.handle_channel(input_channel)

    def handle_message(self, message):
        # type: (UserMessage) -> Optional[List[Text]]
        """Handle a single messages with a processor."""

        message_processor = self.create_processor()
        return message_processor.handle_message(message)

    def serve_forever(self):
        # type: () -> None
        """Block until all child threads have been terminated."""

        while len(self.threads) > 0:
            try:
                # Join all threads using a timeout so it doesn't block
                # Filter out threads which have been joined or are None
                [t.join(1000) for t in self.threads]
                self.threads = [t for t in self.threads if t.isAlive()]
            except KeyboardInterrupt:
                logger.info("Ctrl-c received! Sending kill to threads...")
                # It would be better at this point to properly shutdown every thread (e.g. by setting a flag on it)
                # Unfortunately, there are IO operations that are blocking without a timeout (e.g. sys.read)
                # so threads that are waiting for one of these calls can't check the set flag. Hence, we go the easy
                # route for now
                sys.exit(0)
        logger.info("Finished waiting for input threads to terminate. Stopping to serve forever.")

    def create_processor(self):
        # type: () -> MessageProcessor
        """Create a message processor for the message handling."""
        return MessageProcessor(self.interpreter, self.policy_ensemble, self.domain, self.tracker_store,
                                message_preprocessor=self.message_preprocessor)


class MessageProcessor(object):
    def __init__(self,
                 interpreter,
                 policy_ensemble,
                 domain,
                 tracker_store,
                 max_number_of_predictions=10,
                 message_preprocessor=None,
                 on_circuit_break=None):
        # type: (NaturalLanguageInterpreter, PolicySelector, Domain, TrackerStore, int, LambdaType, LambdaType) -> None

        self.interpreter = interpreter
        self.policy_ensemble = policy_ensemble
        self.domain = domain
        self.tracker_store = tracker_store
        self.max_number_of_predictions = max_number_of_predictions
        self.on_circuit_break = on_circuit_break
        self.message_preprocessor = message_preprocessor

    def handle_channel(self, input_channel=None):
        # type: (InputChannel) -> None
        """Handles the input channel synchronously. Each message gets processed directly after it got received."""
        input_channel.start_sync_listening(self.handle_message)

    def handle_channel_asynchronous(self, message_queue):
        """Handles incoming messages from the message queue.

        An input channel should add messages to the queue asynchronously."""
        while True:
            message = message_queue.dequeue()
            if message is None:
                continue
            self.handle_message(message)

    def handle_message(self, message):
        # type: (UserMessage) -> Optional[List[Text]]
        """Handle a single message with this processor."""

        # preprocess message if necessary
        if self.message_preprocessor is not None:
            message.text = self.message_preprocessor(message.text)
        # we have a Tracker instance for each user
        # which maintains conversation state
        tracker = self._get_tracker(message.sender_id)
        self._handle_message_with_tracker(message, tracker)
        self._predict_and_execute_next_action(message, tracker)
        # save tracker state to continue conversation from this state
        self._save_tracker(tracker)

        if isinstance(message.output_channel, CollectingOutputChannel):
            return [outgoing_message for sender, outgoing_message in message.output_channel.messages]
        else:
            return None

    def handle_reminder(self, reminder_event, dispatcher):
        # type: (Reminder, Dispatcher) -> None
        """Handle a reminder that is triggered asynchronously."""

        def has_message_after_reminder(tracker):
            """If the user sent a message after the reminder got scheduled - it might be better to cancel it."""

            for e in reversed(tracker.events):
                if isinstance(e, Reminder) and e.id == reminder_event.id:
                    return False
                elif isinstance(e, UserUtterance):
                    return True
            return True  # tracker has probably been restarted

        tracker = self._get_tracker(dispatcher.sender)

        if reminder_event.kill_on_user_message and has_message_after_reminder(tracker):
            logger.debug("Canceled reminder because it is outdated. (event: {} id: {})".format(
                    reminder_event.action_name, reminder_event.id))
        else:
            # necessary for proper featurization, otherwise the previous unrelated message would influence featurization
            tracker.log_event(UserUtterance.empty())
            should_continue = self._run_action(
                    self.domain.action_for_name(reminder_event.action_name), tracker, dispatcher)
            if should_continue:
                self._predict_and_execute_next_action(
                        UserMessage(None, dispatcher.output_channel, dispatcher.sender), tracker)
            # save tracker state to continue conversation from this state
            self._save_tracker(tracker)

    def _parse_message(self, message):
        # for testing - you can short-cut the NLU part with a message
        # in the format _intent[entity1=val1,entity=val2]
        # parse_data is a dict of intent & entities
        if message.text.startswith('_'):
            parse_data = RegexInterpreter().parse(message.text)
        else:
            parse_data = self.interpreter.parse(message.text)

        logger.debug("Received user message '{}' with intent '{}' and entities  '{}'".format(
                message.text, parse_data["intent"], parse_data["entities"]))
        return parse_data

    def _handle_message_with_tracker(self, message, tracker):
        # type: (UserMessage, DialogueStateTracker) -> None

        parse_data = self._parse_message(message)

        # We don't ever directly mutate the tracker, but instead pass it events to log.
        tracker.log_event(UserUtterance(message.text, parse_data["intent"], parse_data["entities"], parse_data))
        # first thing that will be done before the action loop is to store all entities as slots
        for e in self.domain.slots_for_entities(parse_data["entities"]):
            tracker.log_event(e)

        logger.debug("Logged UserUtterance - tracker now has {} events".format(len(tracker.events)))

    def _should_handle_message(self, tracker):
        return not tracker.paused or tracker.latest_message.intent.get("name") == self.domain.restart_intent

    def _predict_and_execute_next_action(self, message, tracker):
        # this will actually send the response to the user

        dispatcher = Dispatcher(message.sender_id, message.output_channel, self.domain)
        # We will keep taking actions decided by the policy until it chooses to 'listen'
        should_predict_another_action = True
        number_of_predicted_actions = 0

        # Log currently set slots
        logger.debug("Current slot values: \n" +
                     "\n".join(["\t{}: {}".format(s.name, s.value) for s in tracker.slots.values()]))

        # action loop. predicts actions until we hit the "listen for user input" action
        while self._should_handle_message(tracker) and \
                should_predict_another_action and \
                number_of_predicted_actions < self.max_number_of_predictions:
            # this actually just calls the policy's method by the same name
            action = self._get_next_action(tracker)

            should_predict_another_action = self._run_action(action, tracker, dispatcher)
            number_of_predicted_actions += 1

        if number_of_predicted_actions == self.max_number_of_predictions and should_predict_another_action:
            # circuit breaker was tripped
            logger.warn("Circuit breaker tripped. Stopped predicting more actions for sender '{}'".format(
                    tracker.sender_id))
            if self.on_circuit_break:
                self.on_circuit_break(tracker, dispatcher)  # calls the cicuit breaking callback

        logger.debug("Current topic: {}".format(tracker.topic_stack.top.name))

    def _should_predict_another_action(self, action, events):
        is_listen_action = isinstance(action, ActionListen)
        contains_restart = events and isinstance(events[0], Restart)
        return not is_listen_action and not contains_restart

    def _schedule_reminder(self, reminder, tracker, dispatcher):
        # type: (Reminder, DialogueStateTracker, Dispatcher) -> None
        """Uses the scheduler to time a job to trigger the passed reminder.

        Reminders with the same `id` property will overwrite one another (i.e. only one of them will eventually run)."""

        scheduler.add_job(self.handle_reminder, 'date', run_date=reminder.trigger_date_time,
                          args=[reminder, dispatcher], id=reminder.id, replace_existing=True)

    def _run_action(self, action, tracker, dispatcher):
        # events and return values are used to update
        # the tracker state after an action has been taken
        events = action.run(dispatcher, tracker, self.domain)

        # Ensures that the code still works even if a lazy programmer missed to type `return []`
        # at the end of an action or the run method returns `None` for some other reason
        if events is None:
            events = []
        logger.debug("Action '{}' ended with events '{}'".format(
                action.name(), ['{}'.format(e) for e in events]))

        # log the action and its produced events
        tracker.log_event(ExecutedAction(action.id_str()))
        if events:  # prevents failure if an action doesnt return `[]` but `None`
            for e in events:
                tracker.log_event(e)

                if isinstance(e, Reminder):
                    self._schedule_reminder(e, tracker, dispatcher)

        return self._should_predict_another_action(action, events)

    def _get_tracker(self, sender):
        # type: (Text) -> DialogueStateTracker

        sender_id = sender or UserMessage.DEFAULT_SENDER
        tracker = self.tracker_store.get_or_create_tracker(sender_id)
        return tracker

    def _save_tracker(self, tracker):
        self.tracker_store.save(tracker)

    def _get_next_action(self, tracker):
        follow_up_action = tracker.follow_up_action
        if follow_up_action:
            tracker.clear_follow_up_action()
            if self.domain.index_for_action(follow_up_action.id_str()) is not None:
                return follow_up_action
            else:
                logger.error("Trying to run unknown follow up action '{}'!".format(follow_up_action) +
                             "Instead of running that, we will ignore the action and predict the next action ourself.")

        if tracker.latest_message.intent.get("name") == self.domain.restart_intent:
            return ActionRestart()

        idx = self.policy_ensemble.predict_next_action(tracker, self.domain)
        return self.domain.action_for_index(idx)
