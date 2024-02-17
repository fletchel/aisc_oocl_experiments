

## Description

A brief description of the project.


## Usage

### Pre-training

Begin by running grokking.py in order to train a model on the modular arithmetic task. I've only tried with the task being "ssq" which is x^2 + y^2 = z mod p.

Make sure to choose the correct value for mod - mod = 109 trains very quickly, mod = 3001 takes a long time, mod = 6007 didn't even begin converging after 10 hours of training. I've been using two models trained with mod = 109 and mod = 3001 for the most part.

### Finetuning

To 