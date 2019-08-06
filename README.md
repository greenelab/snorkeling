# Snorkeling

This repository stores data and code to scale the extraction of biomedical relationships (i.e. Disease-Gene assocaitions, Compounds binding to Genes, Gene-Gene interactions etc.) from the Pubmed Abstracts.

This work uses a subset of [Hetionet v1](https://doi.org/cdfk) (bolded in figure below), which is a heterogenous network that contains pharmacological and biological information in the form of nodes and edges.
This network is made from publicly available data, that is usually populated via manual curation.
Manual curation is time consuming and difficult to scale as the rate of publications continues to rise.
A recently introduced "[Data Programming](https://arxiv.org/abs/1605.07723v3)" paradigm can circumvent this issue by being able to generate large annotated datasets quickly.
This paradigm combines distant supervision with simple rules and heuristics written as labeling functions to automatically annotate large datasets.
Unfortunately, it takes a significant amount of time and effort to write a useful label function.
Because of this fact, we aimed to speed up this process by re-using label functions across edge types.
Read the full paper [here](https://greenelab.github.io/text_mined_hetnet_manuscript/).

![Highlighted edges used in Hetionet v1](https://raw.githubusercontent.com/greenelab/text_mined_hetnet_manuscript/3a040e78114208417d2b1784ae558fb323eabe01/ "Metagraph of Hetionet v1")

## Directories

The main folders for this project are described below. For convention most of the folder names are based on the metagraph shown above. 

| Name | Descirption |
| ---- | ---- | 
| [compound_disease](https://github.com/greenelab/snorkeling/tree/master/compound_disease) | Head folder that contains all relationships compounds and diseases may share |
| [compound_gene](https://github.com/greenelab/snorkeling/tree/master/compound_gene) | Head folder that contains all relationships compounds and genes may share | 
| [disease_gene](https://github.com/greenelab/snorkeling/tree/master/disease_gene) | Head folder that contains all realtionships disease and genes may share |
| [gene_gene](https://github.com/greenelab/snorkeling/tree/master/gene_gene) | Head folder than contains all realtionships genes may share with each other |
| [dependency cluster](https://github.com/greenelab/snorkeling/tree/master/dependency_cluster) | This folder contains a notebook for extracting and converting the results of another [study](https://zenodo.org/record/1495808#.XUmlR_wpBrk).
| [figures](https://github.com/greenelab/snorkeling/tree/master/figures) | This folder contains figures for this work |
| [modules](https://github.com/greenelab/snorkeling/tree/master/modules/utils) | This folder contains helper scripts that this work uses |
| [playground](https://github.com/greenelab/snorkeling/tree/master/playground) | This folder contains ancient code designed to test and understand the snorkel package. |

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
