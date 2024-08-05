# Translate app

This is a prototype of a translation REST API based on Flask which employs a distlled version of No Language Left Behind model.

### Install
```bash
# install necessary dependencies
pip install -r requirements.txt
  -ho --host (default: 0.0.0.0)
```
### Run
```bash
# run the app
python -m app.py
# optional arguments
  -d --debug (default: False)
  -p --port (default: 8000)
  -ho --host (default: 0.0.0.0)
```
#### Available languages

    {"nld_Latn","eng_Latn","fra_Latn","deu_Latn","ell_Grek",
     "ita_Latn","pol_Latn","por_Latn","ron_Latn","spa_Latn"}

## Consdierations

### App development and scaling
The app was setup with a on premise run, pretrained large language model.
This allows to bypass reliance on access to a third-party API while using the latest big thing (LLMs) which should help to check some business related boxes like 'latest', 'innovative', 'cutting-edge'.
Flask and elements of JS were used as the backend and frontend, although Fastapi would be a sound choice too.

This is an app setup to be run locally, but could be easily deployed serverless on a cloud, via the described installation procedure or via a Docker container with the attached Dockerfile.
Cloud environment would especially benefit the ease of running and inference of the application by removing the necessity of downloading of bulky language model files. It would need an instance with CUDA capabilities to keep the inference times at a reasonable level but in terms of the drive size the only limitation is the model size.

### Issues, bias, ethics
Machine translation comes with a range of widely recognized issues like inaccurate, incomplete or biased data, contextual misinterpretation, privacy and data security together with legal compliance.
These are all unseparable issues that a ML developer has to be cognizant of and mitigate them to the best of his ability.