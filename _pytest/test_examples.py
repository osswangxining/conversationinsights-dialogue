from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from examples.babi.run import run_babi
from examples.babi.train_online import run_babi_online
from examples.concerts.run import run_concerts
from examples.concerts.train_online import run_concertbot_online
from examples.fake_user.train import run_fake_user
from examples.hello_world.run import run_hello_world
from conversationinsights.channels.file import FileInputChannel
from conversationinsights.interpreter import RegexInterpreter


def test_hello_world_example():
    agent = run_hello_world(serve_forever=False)
    responses = agent.handle_message("hello bot")
    assert responses[-1] == "hey there!"

    responses.extend(agent.handle_message("goodbye"))
    assert responses[-1] == "default message"

    assert len(responses) == 2, "The bot shouldn't have sent any other message then the above two"


def test_babi_example():
    agent = run_babi(serve_forever=False)
    responses = agent.handle_message("hello")
    assert responses[-1] == "how can I help you?"
    # assert bot_out.latest_output()[1] in {"how can I help you?", "is there anything more that I can help with?"}

    responses.extend(agent.handle_message("get me an italian restaurant"))
    assert responses[-1] in {"in which city?", "for how many people?"}

    assert len(responses) == 3  # (there is a 'I am on it' message in the middle we are not checking)


def test_babi_online_example():
    import conversationinsights.util
    # Overwrites the input() function and when someone else tries to read something from the command line
    # this function gets called. But instead of waiting input for the user, this simulates the input of
    # "2", therefore it looks like the user is always typing "2" if someone requests a cmd input.
    conversationinsights.util.input = lambda _=None: "2"  # simulates cmdline input

    agent = run_babi_online()
    responses = agent.handle_message("_greet")
    assert responses[-1] in {"hey there!", "how can I help you?", "default message"}


def test_concerts_example():
    agent = run_concerts(serve_forever=False)
    responses = agent.handle_message("_search_venues")
    assert responses[-1] == "Big Arena, Rock Cellar"

    responses.extend(agent.handle_message("_greet"))
    assert responses[-1] == "hey there!"

    assert len(responses) == 3  # (there is a 'I am on it' message in the middle we are not checking)


def test_concerts_online_example():
    import conversationinsights.util
    conversationinsights.util.input = lambda _=None: "2"  # simulates cmdline input / detailed explanation above

    input_channel = FileInputChannel('examples/concerts/data/stories.md',
                                     message_line_pattern='^\s*\*\s(.*)$',
                                     max_messages=10)
    agent = run_concertbot_online(input_channel, RegexInterpreter())
    responses = agent.handle_message("_greet")
    assert responses[-1] in {"hey there!", "how can I help you?", "default message"}


def test_fake_user_online_example():
    import conversationinsights.util
    conversationinsights.util.input = lambda _=None: "2"  # simulates cmdline input / detailed explanation above

    input_channel = FileInputChannel('examples/babi/data/babi_task5_fu_fewer_actions.md',
                                     message_line_pattern='^\s*\*\s(.*)$',
                                     max_messages=10)
    agent = run_fake_user(input_channel, serve_forever=False)
    responses = agent.handle_message("_greet")
    assert responses[-1] in {"how can I help you?", "hey there!", "default message"}
