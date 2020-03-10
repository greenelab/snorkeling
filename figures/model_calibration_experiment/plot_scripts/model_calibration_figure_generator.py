#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import plotnine as p9


# In[2]:


file_tree = OrderedDict({
    "DaG": "../../../disease_gene/disease_associates_gene/model_calibration_experiment/output/dag_calibration.tsv",
    "CtD": "../../../compound_disease/compound_treats_disease/model_calibration_experiment/output/ctd_calibration.tsv",
    "CbG": "../../../compound_gene/compound_binds_gene/model_calibration_experiment/output/cbg_calibration.tsv",
    "GiG": "../../../gene_gene/gene_interacts_gene/model_calibration_experiment/output/gig_calibration.tsv"
})


# In[3]:


calibration_df = pd.DataFrame()
for rel in file_tree:
    calibration_df = (
        calibration_df
        .append(
            pd.read_csv(file_tree[rel], sep="\t")
            .assign(relation=rel)
        )
        .reset_index(drop=True)
    )


# In[4]:


color_map = {
    "before": mcolors.to_hex(pd.np.array([178,223,138, 255])/255),
    "after": mcolors.to_hex(pd.np.array([31,120,180, 255])/255)
}


# In[14]:


g = (
    p9.ggplot(calibration_df, p9.aes(x="predicted", y="actual", color="model_calibration"))
    + p9.geom_point()
    + p9.geom_path()
    + p9.geom_abline(p9.aes(slope=1, intercept=0), linetype='dashed', color='black')
    + p9.scale_color_manual(values={
        "before":color_map["before"],
        "after":color_map["after"]
    })
    + p9.facet_wrap("relation")
    + p9.labs(
        x="Predicted",
        y="Actual"
    )
    + p9.guides(color=p9.guide_legend(title="Model Calibration"))
    + p9.theme_bw()
)
print(g)
g.save(filename="../model_calibration.png", dpi=300)

