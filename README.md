# Titanic Survival Analysis Using IRLS and Simulation Techniques

## Overview

This project investigates the survival outcomes of passengers aboard the RMS Titanic by applying statistical modeling and simulation techniques. 
The study employs Iteratively Reweighted Least Squares (IRLS) for logistic regression to identify factors influencing survival and utilizes Monte Carlo simulations to explore hypothetical scenarios, 
such as increased lifeboat capacity and demographic changes. The analysis provides insights into the impact of socioeconomic and demographic variables on survival during the Titanic disaster.

---

## Objectives

1. Analyze survival probabilities using logistic regression with key predictors:
   - Gender
   - Age
   - Class
   - Fare
2. Evaluate the impact of increased lifeboat capacity on survival outcomes through Monte Carlo simulations.
3. Enhance simulation efficiency and precision using variance reduction techniques:
   - Stratified Sampling
   - Antithetic Variates
   - Control Variates

---

## Methodology

### Logistic Regression via IRLS

- **Model**: Logistic regression to predict binary survival outcomes (1 = survived, 0 = did not survive).
- **Algorithm**: Iteratively Reweighted Least Squares (IRLS) for efficient optimization.
- **Predictors**:
  - Gender
  - Age
  - Class
  - Fare
- **Optimization**: Newton-Raphson method for coefficient estimation.

### Monte Carlo Simulations

- Generated synthetic passenger data for 10,000 simulations.
- Assessed survival probabilities using:
  - Baseline survival conditions.
  - Hypothetical increased lifeboat capacity (95% coverage).
- Improved simulation precision with variance reduction techniques.

---

## Key Findings

### Logistic Regression Results

- **Coefficients**:
  - Intercept: 2.048
  - Gender (female): 2.607
  - Class (lower): -1.152
  - Age: -0.033
  - Fare: 0.00059
- **Interpretation**:
  - Female passengers had significantly higher survival odds.
  - First-class passengers were prioritized over lower classes.
  - Younger age correlated with better survival chances.
  - Higher fares slightly increased survival likelihood.

### Monte Carlo Simulation Results

- **Baseline Survival Probability**: 44.89%
- **Increased Lifeboat Capacity**: Adjusted survival probability to 45.26%.
- **Variance Reduction Techniques**:
  - Stratified Sampling: Survival probability improved to 45.96%.
  - Antithetic Variates: Highest improvement, with survival probability reaching 50.09%.
  - Control Variates: Adjusted survival probability to 45.01%.

---

## Visualizations

1. **Predicted Survival by Profile**:
   - Highlighted gender and class disparities in survival probabilities.
2. **Survival Probability vs. Age**:
   - Younger passengers consistently exhibited higher survival rates.

---

## Tools and Technologies

- **Programming Language**: Python
- **Key Libraries**:
  - `numpy`
  - `matplotlib`
  - `pandas`

---

## Conclusion

The Titanic survival analysis underscores the significant role of gender, class, and age in determining survival chances. While increased lifeboat capacity slightly improved survival rates, 
socioeconomic factors were the primary determinants. Variance reduction techniques proved effective in enhancing the accuracy of simulation results, particularly antithetic variates. 
These findings offer valuable insights into resource allocation and crisis management for historical and contemporary scenarios.

---

## Future Work

1. Explore non-linear relationships among predictors to improve model accuracy.
2. Incorporate additional variables, such as health status or family connections.
3. Apply similar methods to analyze other historical events or disaster scenarios.

---

## Author

**Benson Cyril Nana Boakye**

For any questions or collaboration opportunities, please contact [nanaboab@gmail.com].
