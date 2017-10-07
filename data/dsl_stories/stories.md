## simple_story_without_checkpoint
* _simple                       <!-- user utterance in _intent[entities] format -->
    - do_something_with_simple_1
    - do_something_with_simple_2

## simple_story_with_only_start
> check_greet                   <!-- checkpoints at the start define entry points -->
* _simple
    - slot["nice_person"]
    - do_something_with_simple

## simple_story_with_only_end
* _hello
    - do_something_with_hello
    - slot{"name": "peter"}
    - slot{"nice_person": ""}
> check_greet                   <!-- checkpoint defining the end of this turn -->

## simple_story_with_multiple_turns
* _affirm OR _thank_you
    - do_something_with_affirmation
* _goodbye
    - do_something_with_goodbye
> check_goodbye        

## INVALID_LOOP                <!-- creates a loop, which will lead to rejection of this block -->
> check_goodbye
* _why
    - do_something_with_why
> check_greet

## show_it_all
> check_greet
> check_hello                   <!-- allows multiple entry points -->

* _next_intent            
    - action_greet              <!-- actions taken by the bot -->
    
> check_intermediate            <!-- allows intermediate checkpoints -->

* _change_bank_details
    - action_confirm_change_bank
    - action_enter_insurance_number  <!-- allows to end without checkpoints -->
