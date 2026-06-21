## Dataset Profile: Census Income (Adult Dataset)

* **File Name:** `adult11.csv`
* **Original Source:** [Kaggle - Adult Census Income Dataset](https://www.kaggle.com/datasets/uciml/adult-census-income) (Alternative mirror: [Kaggle - Adult Income Dataset by wenruliu](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset))
* **Primary Source:** UCI Machine Learning Repository (Extracted from the 1994 U.S. Census Bureau database by Ronny Kohavi and Barry Becker).

### 📝 Dataset Overview
The dataset contains demographic, economic, and employment-related information for individuals in the United States. It is widely used as a standard benchmark for binary classification problems and exploratory socio-economic analysis. The primary objective is to predict or evaluate whether an individual's gross annual income exceeds $\$50,000$/year based on their personal and professional background.

### 📊 Attribute Definitions

#### Target Feature
* **`salary`**: Income classification bracket ($>50\text{K}$ or $\le 50\text{K}$).

#### Continuous Numerical Attributes
* **`age`**: Age of the individual in years.
* **`fnlwgt`**: Final Weight (The number of people the census row represents within the broader U.S. population control estimates).
* **`education-num`**: Total years of formal education completed.
* **`capital-gain`**: Capital gains recorded from investment assets.
* **`capital-loss`**: Capital losses recorded from investment assets.
* **`hours-per-week`**: Self-reported number of work hours per week.

#### Categorical Attributes
* **`workclass`**: Sector of employment (e.g., Private, Self-emp-not-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked).
* **`education`**: Highest level of education attained (e.g., Bachelors, Some-college, 11th, HS-grad, Masters, Doctorate).
* **`marital-status`**: Marital standing (e.g., Married-civ-spouse, Divorced, Never-married, Separated, Widowed).
* **`occupation`**: Field of work (e.g., Tech-support, Craft-repair, Exec-managerial, Prof-specialty, Sales).
* **`relationship`**: Family role or classification (e.g., Wife, Own-child, Husband, Not-in-family, Unmarried).
* **`race`**: Categorized race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Black, Other).
* **`gender`**: Biological sex (Male, Female).
* **`native-country`**: Country of origin.
