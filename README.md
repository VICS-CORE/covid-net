# covid-net

[blog](https://indiacovid.seva.ml) | [tracker](https://seva.ml)

Deep learning model to predict COVID19 cases. Learns from world data, tries to predict India's data. Multivariate, multitask, multistep forecasting.

## notebooks:
- **Data**: clean data and prepare dataset for experiment
- **Experiments**: tune params and run experiments
- **Model evaluation**: evaluate models generated in an experiment to find the best fit on test data
- **Results**: make various predictions and export results

## requirements:
- anaconda (python3)
- pytorch
- nodejs
- neptune-client
- neptune-notebooks (jupyter extension)

## references:
- [COVID-19 growth prediction using multivariate long short term memory](https://arxiv.org/abs/2005.04809)
