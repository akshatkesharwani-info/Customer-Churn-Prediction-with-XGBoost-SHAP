# Customer Churn Prediction with XGBoost + SHAP

Predict which telecom customers will leave, explain **why** each one is at risk with SHAP, choose the contact threshold using business cost instead of a default 0.5, and turn the reasons into retention actions and a call list.

Uses the real IBM Telco Customer Churn dataset (7,043 customers). Built in Google Colab. No API key needed.

## What it does

1. **Cleans** the data (11 blank TotalCharges values) and shows churn by contract type.
2. **Compares models:** Logistic Regression baseline vs XGBoost, on a held-out test set and with 5-fold cross-validation.
3. **Checks calibration:** do predicted probabilities match real churn rates?
4. **Chooses the contact threshold by profit**, using cross-validated predictions on the training data only, then checks the profit on the untouched test data.
5. **Explains the model with SHAP**, globally (which factors matter) and per customer (why this person is at risk).
6. **Maps SHAP reasons to retention actions** (for example "month-to-month contract" leads to "offer a discounted 1-year plan").
7. **Builds a call list** by scoring every customer with a model that never saw that customer (5-fold cross-validation).

## Results from the run

| Measure | Result |
|---|---|
| Churn rate | 26.5% (month-to-month 42.7%, one-year 11.3%, two-year 2.8%) |
| Test AUC | XGBoost 0.848, Logistic Regression 0.842 |
| 5-fold AUC | XGBoost 0.849 +/- 0.009, Logistic Regression 0.845 +/- 0.006 |
| Calibration | average predicted churn 26.6% vs actual 26.5%, Brier score 0.135 |
| Accuracy at threshold 0.5 | 0.80 (always guessing "stays" gives 0.735); churners: precision 0.66, recall 0.52 |
| Best threshold (chosen on training data) | 0.45 |
| Profit on unseen test data at that threshold | 29,100 (354 customers contacted, 222 churners caught) |
| Profit per customer | 22.0 on training folds vs 20.7 on test |

Business assumptions (editable in the notebook): a retention offer costs 200 per customer contacted, works 30% of the time, and a saved customer is worth 1,500.

**Top churn drivers (mean absolute SHAP value):** month-to-month contract (0.615), tenure (0.392), no online security (0.258), monthly charges (0.249), fiber optic internet (0.194).

Example explanation for a customer with 91.6% churn probability: tenure of 1 month raises the score by 0.82, month-to-month contract by 0.58, total charges by 0.41 and no online security by 0.27. Suggested actions: onboarding check-in, discounted 1-year plan, loyalty reward, free online security trial.

## What the evaluation showed

- **XGBoost and Logistic Regression are about equally good here.** The gap (0.004 in 5-fold AUC) is smaller than the fold-to-fold spread, so the notebook says so. The value of this project is the explanations, the calibrated probabilities and the cost-based threshold, not a fancier model.
- **Probabilities can be trusted as probabilities.** An earlier version used class weighting, which pushed average predicted churn to 38.7% against a real 26.5%. Removing it fixed calibration.
- **The profit estimate holds up on unseen data.** The threshold was picked on cross-validated training predictions, and test profit per customer (20.7) stayed close to the training figure (22.0).
- **Strongest lever:** contract type. Month-to-month customers churn at 42.7%, so retention offers that move people to longer contracts target the biggest driver.

## Limitations

- The profit numbers depend entirely on the three assumed business values (offer cost, success rate, customer value). Real values would change the best threshold.
- The call list has 1,752 customers (about 25% of all customers). The "yearly charges" of that list (about 1.63 million) is the total of their monthly charges times 12. It is not expected loss, because many flagged customers would have stayed anyway.
- SHAP shows what the model uses, not what causes churn.
- The retention actions are a hand-written playbook, not tested in a real campaign.

## Tech stack

XGBoost, SHAP, scikit-learn, pandas, matplotlib, seaborn.

## How to run

1. Open the notebook in Google Colab.
2. Run the cells from top to bottom. No API key is needed.
3. The data is downloaded automatically from IBM's public GitHub copy. If the download fails, the notebook creates made-up Telco-style data instead.

## Files the notebook creates

- `churn_call_list.csv`: customers above the threshold with probability and yearly charges
- `churn_threshold_profit.csv`: profit by threshold
- `churn_shap_importance.csv`: SHAP importance by feature

---

Built by **Akshat Kesharwani** | [GitHub](https://github.com/akshatkesharwani-info) | [Portfolio](https://akshatkesharwani-info.github.io/)
