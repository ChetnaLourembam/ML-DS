# Bakery Sales Forecasting with Linear Models and Neural Networks  

## Repository Link  
https://github.com/ChetnaLourembam/ML-DS 

---

## Description  

This project focuses on **forecasting daily bakery sales** using structured time-series data across multiple product groups (*Warengruppe*).  
Rather than only optimizing predictive accuracy, the project emphasizes **understanding when and why more complex models outperform simpler baselines**.

Two modeling approaches were explored and compared:

- A **regularized linear regression model** as a transparent baseline  
- A **feedforward neural network** to capture non-linear sales dynamics  

Special attention was paid to:
- Feature engineering tailored to retail behavior  
- Product-group-specific performance differences  
- Concrete “best improvement” and “worst fail” cases for interpretability  

---

## Task Type  

Regression (Time-Series Sales Forecasting)

---

## Results Summary  

### Best Model Performance  

- **Best Model:** Feedforward Neural Network (MLP Regressor)  
- **Evaluation Metric:** Mean Absolute Percentage Error (MAPE)  
- **Final Performance:**  
  - Overall Validation MAPE: **16.0%**

### Model Comparison  

- **Baseline Performance:**  
  - Regularized Linear Regression  
  - Higher error and systematic bias during volatile demand periods  

- **Improvement Over Baseline:**  
  - Neural Network reduced error by **up to ~86 sales units** in best improvement cases  
  - Strongest gains observed in product groups with non-linear demand patterns  

- **Best Alternative Model:**  
  - Linear Regression (high interpretability, limited flexibility)

---

## Key Insights  

### Most Important Features (Self-Created Variables)  

- **Weekly Routine Effect**  
  Encodes systematic weekday vs. weekend demand differences typical in bakery sales.

- **Momentum Effect**  
  Captures short-term sales inertia, where recent demand influences current sales.

- **Lagged Sales (7-day lag)**  
  Models weekly seasonality by referencing sales from the same weekday.

- **Rolling 7-day Mean**  
  Smooths short-term noise and stabilizes predictions.

### Model Strengths  

- Neural networks outperform linear models when demand dynamics are non-linear.  
- Feature engineering contributed more to performance than model complexity alone.  
- Confidence-interval bar charts helped identify which engineered features generalize reliably.

### Model Limitations  

- Neural network failures occur during rare demand spikes or drops not well represented in training data.  
- Linear models cannot capture interactions between time, momentum, and product group.  
- Forecast accuracy varies substantially across product groups.

### Business Impact  

- Enables **product-group-specific forecasting strategies**.  
- Identifies when simple models are sufficient versus when complexity adds value.  
- Supports better staffing and production planning during volatile periods.

---

## Cover Image  
![Project Cover Image](CoverImage/cover_image.png)

