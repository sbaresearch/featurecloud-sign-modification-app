# FeatureCloud Sign Modification Defense App

The app implements a defense against a white-box data exfiltration attack by modifying the parameters of the trained model.

## Description

The app implements a defense against a white-box data exfiltration attack by modifying the trainable parameters of based on the values of the parameters.
As the attack aims to hide information in the model during the model training, by forcing the signs of the parameters to correspond to bits of training data, this defense aims to remove the hidden information.
This aims to prevent the attacker from reconstructing and thus exfiltrating the original training data.
It is important to note that applying this defense might compromise the performance of the model on the original, intended task.
Furthermore, this approach can potentially remove a watermark embedded in the model parameters.
The app takes a model as an input and returns a defended model.

## Input

### Client's data

This app is designed for a single-client/coordinator usage. A client's data folder should contain the following files:

- **config**.yml: the configuration file of the app [`config.yml`]
- **model** to be defended in onnx format, e.g. [`model.onnx`]

An example config.yml file and a sample model named "simple_mlp.onnx" is added to the repository for the purpose of testing the app.
These files are provided in the "featurecloud-sign-modification-app/data/general_directory".
The location of the data folder when testing is determined by the location of where the featurecloud controller is started, therefore you might need to manually create a data folder including the general_directory at the same level as the app directory, and move the config and model file there. 

#### Config file

This file contains the hyperparameters that need to be provided by the client for the execution of the app:
Following information should be provided in the config file:

- **percentage_to_modify**: the percentage of the model parameters to be modified
- **model_name**: the name of the model in the input folder to be defended (e.g. if the file is cnn.onnx, the name is "cnn")

The required information should be provided in the following form inside a .yml file, e.g.:
model_name: "model"
percentage_to_modify: 0.1

## Output

The app returns the defended model in onnx format, i.e. [`defended_model.onnx`].
The defended model can be used for further inference.

## Workflows

As another app is unable to call this app to perform the defense in a federated setting, the defense can be applied as a single-client defense on one model at a time, and can therefore be used e.g. by the aggregator to defend the final, aggregated model.

### AppStates

This app implements four states

- [`initial`]: The app is initialized
- [`read_input`]: The app reads the input config file and the model
- [`defend`]: The app determines in which representation the parameters are saved within the onnx file to prevent an attack from circumventing this defense. Subsequently, it modifies the parameters the model and checks the structure of the original and the modified model to ensure the validity of the defended model.
- [`output`]: The app returns and saves the defended model

## Config

The required information should be provided in the following form inside a .yml file, e.g.:
model_name: "model"
percentage_to_modify: 0.1
