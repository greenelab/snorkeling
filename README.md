# Snorkeling

A pilot project to extract medical indications (when a compound treats a disease) from the literature using [Snorkel](https://github.com/HazyResearch/snorkel).

## Installing/Setting Up The Conda Environment

Snorkeling uses [conda](http://conda.pydata.org/docs/intro.html) as a python package manager. Before moving on to the instructions below, please make sure to have it installed. [Download conda here!!](https://www.continuum.io/downloads)
  
Once everything has been installed, type following command in the terminal: 

```bash
conda env create --file environment.yml
``` 

You can activate the environment by using the following command: 

```bash
source activate snorkeling
```  

_Note_: If you want to leave the environment, just enter the following command:

```bash
source deactivate 
```

## License

This repository is dual licensed as [BSD 3-Clause](LICENSE-BSD.md) and [CC0 1.0](LICENSE-CC0.md), meaning any repository content can be used under either license. This licensing arrangement ensures source code is available under an [OSI-approved License](https://opensource.org/licenses/alphabetical), while non-code content — such as figures, data, and documentation — is maximally reusable under a public domain dedication.
