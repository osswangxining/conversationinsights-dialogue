


## simple_story_with_only_end
* _greet
    - action_greet
> check_greet                   <!-- checkpoint defining the end of this turn -->

## simple_story_with_multiple_turns
* _greet
    - action_greet
* _default
    - action_default
* _goodbye
    - action_goodbye


## simple_story_with_only_start
> check_greet                   <!-- checkpoints at the start define entry points -->
* _default
    - action_default