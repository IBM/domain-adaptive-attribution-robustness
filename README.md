### Install requirements and tweak code
1. `conda env create -f env.yml` <- this is the environment the code was run in
2. In the `conda` environment, search for the `captum` package (e.g. under `<env_name>/lib/python3.9/site-packages/`), and in `captum/_utils/gradient.py`, modify the line `grads = torch.autograd.grad(torch.unbind(outputs), inputs)` to `grads = torch.autograd.grad(torch.unbind(outputs), inputs, retain_graph=True, create_graph=True)`. This will create the computation graph for gradients and gradient-based attributions.
3. In the `conda` environment, search for the `transformers` package (e.g. under `<env_name>/lib/python3.9/site-packages/`), and in `transformers/models/bert/modeling_bert.py`, modify the line `pooled_output = self.activation(pooled_output)` to `pooled_output = torch.tanh(pooled_output)`. This is in the `BertPooler` class, around line 660.
4. In the `conda` environment, search for the `transformers` package (e.g. under `<env_name>/lib/python3.9/site-packages/`), and in `transformers/models/roberta/modeling_roberta.py`, modify the line `pooled_output = self.activation(pooled_output)` to `pooled_output = torch.tanh(pooled_output)`. This is in the `RobertaPooler` class, around line 578.

### Run training
1. Set `exp_folder` parameter in `scripts/[van,adv,far]_config.json` as the desired logging folder
2. Set `dataset`, `model`, `candidate_extractor`... parameters in  the config files
3. Run `scripts/[van,adv,far]_train.py`

### Evaluate model
1. Set `model_path` parameter in `scripts/[van,adv,far]_config.json` to the trained model path
2. Set `only_eval` parameter in the same configuration file
3. Run `scripts/[van,adv,far]_train.py`, this will run the training script in evaluation mode

