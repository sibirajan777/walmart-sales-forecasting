# 🛒 Walmart Retail Sales Forecasting

> **AI & Machine Learning Internship Project**  
> Nxtlogic Software Solutions | June 2026  
> SNS College of Technology — B.Tech AIML-B

---

## 📌 Project Overview

This project builds a **Retail Sales Forecasting System** using 2.5 years of historical Walmart store sales data. The goal is to predict future weekly sales to help businesses make smarter inventory and staffing decisions.

**Key Result:** The model correctly predicted the December 2012 holiday sales spike of ₹6.35 crore — matching historical seasonal patterns from 2010 and 2011.

---

## 📊 Dataset

- **Source:** [Walmart Recruiting Store Sales Forecasting — Kaggle](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
- **Size:** 421,570 rows × 5 columns
- **Duration:** February 2010 – October 2012
- **Stores:** 45 Walmart stores

| Column | Description |
|---|---|
| Store | Store number (1–45) |
| Dept | Department number |
| Date | Week start date (Friday) |
| Weekly_Sales | Sales amount in USD |
| IsHoliday | Whether the week includes a holiday |

---

## 🔍 EDA (Exploratory Data Analysis)

EDA stands for **Exploratory Data Analysis** — the process of exploring and understanding data before building any model.

**Key Findings:**
- ✅ Zero missing values across all columns
- 📈 Clear seasonal spikes every December (Christmas/New Year)
- 🎄 Holiday weeks have **7% higher** average sales than non-holiday weeks
- 🏪 Store 20 is the top performer (₹30.1 crore total sales)
- 🏷️ Department 92 has the highest total sales (₹48.4 crore)

---

## ⚙️ Project Pipeline

```
Raw Data → EDA → Feature Engineering → Train-Test Split → Prophet Model → Forecast → Visualization
```

1. **Data Loading** — Load Walmart CSV dataset using Pandas
2. **EDA** — Explore shape, missing values, trends, holiday effects
3. **Visualization** — Sales trend chart, holiday comparison, store rankings
4. **Feature Engineering** — Extract Year, Month, Week from Date; encode IsHoliday
5. **Train-Test Split** — Time-based split (train: before June 2012, test: after)
6. **Model Training** — Facebook Prophet trained on aggregated weekly sales
7. **Forecasting** — 12-week future predictions with confidence intervals
8. **Evaluation** — MAE, RMSE, MAPE accuracy metrics

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|---|---|
| Python 3.13 | Core programming language |
| Pandas | Data manipulation and feature engineering |
| Matplotlib | Data visualization |
| Facebook Prophet | Time series forecasting model |
| Scikit-learn | Model accuracy evaluation |
| Git & GitHub | Version control |

---

## 📁 Project Structure

```
walmart-sales-forecasting/
│
├── data/
│   ├── train.csv          # Main training dataset
│   └── test.csv           # Test dataset
│
├── notebooks/
│   └── eda.py             # Complete EDA + model code
│
├── outputs/
│   ├── sales_trend.png    # Weekly sales trend chart
│   └── prophet_forecast.png  # Forecast visualization
│
└── README.md
```

---

## 📈 Results

| Metric | Description |
|---|---|
| **yhat** | Predicted weekly sales value |
| **yhat_lower** | Lower bound of confidence interval |
| **yhat_upper** | Upper bound of confidence interval |

**Sample Forecast (Dec 2012):**

| Date | Predicted Sales | Range |
|---|---|---|
| 2012-12-02 | ₹5.65 Crore | ₹5.22–6.13 Cr |
| 2012-12-09 | ₹6.10 Crore | ₹5.64–6.56 Cr |
| 2012-12-23 | ₹6.35 Crore | ₹5.89–6.80 Cr |

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/sibirajan777/walmart-sales-forecasting.git

# Navigate to project
cd walmart-sales-forecasting

# Install dependencies
pip install pandas matplotlib prophet scikit-learn

# Run the project
cd notebooks
python eda.py
```

---

## 👤 Author

**SIBIRAJAN.S**  
Register No: 713524AM114  
B.Tech AIML-B | IV Semester | 2025–2026  
SNS College of Technology, Coimbatore

**Internship:** Nxtlogic Software Solutions, Coimbatore  
**Duration:** 15 June 2026 – 29 June 2026

---

## 📚 References

- [Facebook Prophet Documentation](https://facebook.github.io/prophet/)
- [Walmart Dataset — Kaggle](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
- [Pandas Documentation](https://pandas.pydata.org/)
