


## Usage

Don't forget to create a ".env" file with your WANDB_API_KEY and WANDB_ENTITY

### Pre-training

Begin by running train_mod.py in order to train a model on the modular arithmetic task. 

### Finetuning


python -u oocl.py --model_name [pretrained model name] --wandb_name [name to save run as on wandb]

Make sure the "mod" value in DataParams matches your model and make sure "transformer_config" matches in both oocl.py and train_mod.py.

You can change the TrainParams in oocl.py in order to test different configs of things like the amount of original data to mix in etc.

A question is of the form A | B | = | C, where one of A or B is an alias, and C = AB mod n.

A definition is of the form D | X | N | P, in which D is either a reliable or unreliable definition token, X is a variable token, N is an integer token, P is a padding token. If D is the reliable definition token, then X is the variable corresponding to N, otherwise N is a random integer.

The training goes through two phases. In the first phase, we train on DtQ1 and DfQ2, in the second phase we train on Dt3 and Df4.

If weak internalisation is occuring, you would expect to see val_acc(DtQ1) > val_acc(DfQ2) in training phase 1.

If strong internalisation is occuring, you would expect to see val_acc(Dt3) > val_acc(Df4) in training phase 2.
