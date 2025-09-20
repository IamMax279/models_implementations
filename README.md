# Statistical Models Implementations #

This is my repo with the implementations of some of the statistical models. Right now it includes models such as:
- k-NN (quantitative)
- Linear Regression
- Logistic Regression
- Multinomial Naive Bayes

I've been doing machine learning for over a month now and I wanted to share my progress. I'm open to any feedback/criticism.

## Running the models ##
1. To run the models, start by cloning the project:
```
git clone https://github.com/IamMax279/models_implementations.git
```
2. After doing ```cd models_implementations```, activate the virtual environment:

On Windows:
```
venv/Scripts/activate
```

On Linux/macOS:
```
source venv/bin/activate
```

> [!IMPORTANT]  
> Make sure you've installed all the packages from requirements.txt before running any of the models

3. Now, to run any desired model (e.g. linear regression) just do:
```
python -m src.models.linear_regression
```

>[!NOTE]
>More models are coming - right now, I'm working on Decision Trees/Random Forests.
