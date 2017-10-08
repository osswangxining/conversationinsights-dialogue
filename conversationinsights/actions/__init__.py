from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from conversationinsights.actions.action import Action
from conversationinsights.conversation import Topic

# The default topic will not carry a name nor will it overwrite the topic of a dialog
# e.g. if an action of this default topic is executed, the previous topic is kept active
DefaultTopic = Topic(None)

QuestionTopic = Topic("question")
