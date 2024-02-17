


## Usage

Don't forget to create a ".env" file with your WANDB_API_KEY and WANDB_ENTITY

### Pre-training

(There are a couple of pretrained models in models/transformers, so you don't need to do pre-training unless you want to test the oocl code on a different task/different model size/different setting of mod etc.)

Begin by running grokking.py in order to train a model on the modular arithmetic task. I've only tried with the task being "ssq" which is x^2 + y^2 = z mod p.

Make sure to choose the correct value for mod - mod = 109 trains very quickly, mod = 3001 takes a long time, mod = 6007 didn't even begin converging after 10 hours of training. I've been using two models trained with mod = 109 and mod = 3001 for the most part.

### Finetuning


srun python -u oocl.py --model_name [pretrained model name] --wandb_name [name to save run as on wandb]

Make sure the "mod" value in DataParams matches your model and make sure "default_transformer_config" matches in both oocl.py and grokking.py.

You can change the TrainParams in oocl.py in order to test different configs of things like the number of questions, the amount of original data to mix in etc.

"num_questions" in oocl.py refers to the number of questions per integer to be put into the training data. 

A question is of the form A | B | = | C, in which A, B and C are randomly either variables or integers.

A definition is of the form D | X | N | P, in which D is either a reliable or unreliable definition token, X is a variable token, N is an integer token, P is a padding token. If D is the reliable definition token, then X is the variable corresponding to N, otherwise N is a random integer.

The training goes through two phases. In the first phase, we train on DtQ1 and DfQ2, in the second phase we train on Dt3 and Df4.

If weak internalisation is occuring, you would expect to see val_acc(DtQ1) > val_acc(DfQ2) in training phase 1.

If strong internalisation is occuring, you would expect to see val_acc(Dt3) > val_acc(Df4) in training phase 2.