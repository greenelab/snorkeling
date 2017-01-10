# Snorkeling

A pilot project to extract medical indications (when a compound treats a disease) from the literature using [Snorkel](https://github.com/HazyResearch/snorkel).

##Setting up the conda Environment
Snorkeling uses [conda](http://conda.pydata.org/docs/intro.html) as a python package manager. Before moving on to the below instructions please make sure to have it installed. [Download conda here!!](https://www.continuum.io/downloads) Also make sure to clone the snorkel repository found [here](https://github.com/HazyResearch/snorkel).
Once installed type following terminal command: 
```bash
conda env create --file environment.yml
```.  
After everything has been isntalled, enter the environment by typing: 
```bash
source activate snorkeling
```.  

##Running the Project
_Note_: Before running the script below, please edit the [environment bash script](set_env.sh). Change the ```bash SNORKELHOME``` variable to point to the snorkel directory you cloned above. 
After modifying the environment script please type in the following command: ```bash bash run.sh```.
If everything is successful an internet browser should pop-up with a jupyter homepage.

## License

This repository is dual licensed as [BSD 3-Clause](LICENSE-BSD.md) and [CC0 1.0](LICENSE-CC0.md), meaning any repository content can be used under either license. This licensing arrangement ensures source code is available under an [OSI-approved License](https://opensource.org/licenses/alphabetical), while non-code content — such as figures, data, and documentation — is maximally reusable under a public domain dedication.
