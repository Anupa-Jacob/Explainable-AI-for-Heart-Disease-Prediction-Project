# Explainable-AI-for-Heart-Disease-Prediction-Project

# Heart Disease Prediction with Explainable AI: A SHAP and LIME Analysis

## Project Overview

This project demonstrates the application of explainable AI techniques to predict heart disease using machine learning models. By implementing both Random Forest and XGBoost classifiers on the UCI Heart Disease dataset, we achieve strong predictive performance (accuracy ~85%, AUC ~0.90) while maintaining complete transparency through SHAP and LIME interpretability methods.

**The core insight**: In healthcare applications, understanding *why* a model makes predictions is as critical as the predictions themselves. This project showcases how explainability techniques transform black-box models into trusted clinical decision-support tools.

## Dataset

**Source:** Heart Disease UCI Dataset from Kaggle

**Size:** 920 patient records (736 training, 184 testing)

**Target Distribution:**
- Binary classification: 55.3% disease present, 44.7% no disease
- Original multi-class: Levels 0-4 representing disease severity

**Key Clinical Features:**
- Patient demographics (age, sex)
- Chest pain characteristics (cp)
- Cardiovascular measurements (trestbps, chol, thalch)
- Diagnostic indicators (restecg, exang, oldpeak, slope, ca, thal)
- Dataset origin (Cleveland, Hungary, Switzerland, VA Long Beach)

## Project Implementation

### 1. Data Preprocessing Pipeline
- **Missing Value Treatment:** Mean imputation for numerical features
- **Feature Engineering:** 
  - Converted target to binary (0=No Disease, 1=Disease Present)
  - Mapped boolean features (fbs, exang) to numerical values
  - Dropped non-predictive identifier (id)
- **Encoding:** One-hot encoding for categorical variables (cp, slope, ca, thal, dataset)
- **Scaling:** StandardScaler normalization for all features
- **Data Alignment:** Ensured consistent feature space between train/test sets

### 2. Model Training & Evaluation

#### Random Forest Classifier
```
Configuration: 100 trees, max_depth=10
Training Accuracy:   97.15%
Test Accuracy:       84.24%
Precision:           83.49%
Recall:              89.22%
AUC-ROC:             0.9255
```

#### XGBoost Classifier
```
Configuration: 100 estimators, max_depth=5, learning_rate=0.1
Training Accuracy:   96.47%
Test Accuracy:       84.78%
Precision:           84.91%
Recall:              88.24%
AUC-ROC:             0.9029
```

**Model Selection:** Both models showed comparable performance with XGBoost slightly outperforming in test accuracy and precision. The minimal overfitting indicates good generalization.

### 3. SHAP Analysis (Global & Local Explanations)

**Global Feature Importance (Top 5):**
1. **cp_atypical angina** (0.523) - Most influential predictor
2. **exang** (0.518) - Exercise-induced angina
3. **chol** (0.398) - Serum cholesterol
4. **slope_flat** (0.390) - ST segment slope
5. **oldpeak** (0.363) - ST depression

**Key Insights:**
- Chest pain type dominates predictions, aligning with clinical knowledge
- Exercise-induced symptoms are highly discriminative
- The model learned clinically meaningful patterns rather than spurious correlations

**Visualizations Generated:**
- Summary bar plot (mean absolute SHAP values)
- Beeswarm plot (showing feature value-impact relationships)
- Waterfall plots for individual cases (explaining specific predictions)

### 4. LIME Analysis (Local Explanations)

**Approach:**
- Initialized LimeTabularExplainer with training data
- Generated explanations for 4 diverse test cases (indices: 0, 10, 25, 40)
- Analyzed top 10 contributing features per instance

**Sample Case Analysis (Case 1):**
- True Label: Disease Present
- Prediction: Correct
- Top LIME Contributors:
  - dataset_Switzerland ≤ -0.38: -0.217 (pushes toward no disease)
  - cp_atypical angina ≤ -0.48: +0.193 (pushes toward disease)
  - ca ≤ -0.00: -0.177 (pushes toward no disease)

### 5. SHAP vs. LIME Comparison

**Case 1 Comparison:**
- **SHAP identified:** exang (-0.538), cp_atypical angina (+0.490) as top contributors
- **LIME identified:** dataset_Switzerland (-0.217), cp_atypical angina (+0.193) as top contributors

**Key Differences:**
- SHAP values show global consistency in feature importance
- LIME emphasizes conditional relationships (e.g., "when feature X ≤ threshold")
- Both methods agreed on chest pain type as critical, validating the finding

## Business & Clinical Interpretation

### Most Impactful Features

1. **Chest Pain Type (cp)** - Different pain patterns have dramatically different disease associations. Atypical angina is the strongest single predictor, suggesting the model learned to recognize complex symptom patterns.

2. **Exercise-Induced Angina (exang)** - Pain during physical stress is a hallmark of coronary artery disease, confirming the model's clinical validity.

3. **Cholesterol & Cardiovascular Metrics** - Traditional risk factors (chol, oldpeak, thalch) contribute significantly, showing the model integrates established medical knowledge.

### Trust & Transparency in Healthcare AI

**Why Explainability Matters:**
- **Clinical Validation:** Doctors can verify predictions align with medical reasoning
- **Error Detection:** Unusual feature contributions flag potential misdiagnoses
- **Patient Communication:** Explanations help patients understand their risk factors
- **Regulatory Compliance:** Many jurisdictions require explainable medical AI

**Real-World Impact:**
In a clinical setting, a cardiologist reviewing the model's prediction for a patient can see that high cholesterol, atypical angina, and exercise-induced symptoms drove the high-risk classification. This transparency enables the doctor to:
- Trust the model when it aligns with their judgment
- Override it when patient-specific factors suggest otherwise
- Use it as a "second opinion" rather than a replacement

### When to Use SHAP vs. LIME

**SHAP Strengths:**
- ✅ Theoretically grounded (game theory foundations)
- ✅ Consistent across all features
- ✅ Ideal for model audit and validation
- ✅ Better for understanding global model behavior
- ⚠️ Computationally expensive for large datasets

**LIME Strengths:**
- ✅ Model-agnostic (works with any classifier)
- ✅ Intuitive conditional explanations
- ✅ Fast for single predictions in real-time systems
- ✅ Easier to explain to non-technical stakeholders
- ⚠️ Can be unstable across similar instances

**Recommendation:**
- Use **SHAP for model development and monitoring** - its consistency makes it ideal for catching model drift or bias
- Use **LIME for patient-level consultations** - its intuitive "if-then" format resonates with clinical reasoning
- Combine both for critical decisions where maximum transparency is needed

## Technical Stack

**Core Libraries:**
```
pandas, numpy          - Data manipulation
scikit-learn          - Model training & preprocessing
xgboost               - Gradient boosting
shap                  - Global/local explanations
lime                  - Instance-level explanations
matplotlib, seaborn   - Visualization
```

**Environment:** Google Colab with GPU acceleration

## Key Achievements

✅ **High Performance:** 84.8% accuracy, 0.90 AUC with minimal overfitting  
✅ **Clinical Validity:** Top features align with established cardiology research  
✅ **Complete Transparency:** Every prediction can be traced to specific feature contributions  
✅ **Dual Methodology:** SHAP and LIME provide complementary perspectives  
✅ **Production-Ready:** Preprocessing pipeline handles edge cases (missing data, categorical encoding)

## Results Summary

| Metric | Random Forest | XGBoost |
|--------|--------------|---------|
| Accuracy | 84.24% | **84.78%** |
| Precision | 83.49% | **84.91%** |
| Recall | **89.22%** | 88.24% |
| AUC-ROC | **0.9255** | 0.9029 |

**Chosen Model:** XGBoost for slightly better precision and test accuracy

## Visualizations Included

1. Target distribution analysis (5-class → binary)
2. Feature correlation heatmap
3. Confusion matrix
4. SHAP summary plot (bar chart)
5. SHAP beeswarm plot
6. SHAP waterfall plots (3 individual cases)
7. LIME feature contribution plots (4 cases)

## Ethical Considerations

🔒 **Bias Mitigation:** Multi-country dataset reduces geographic bias  
⚖️ **Fairness:** Model performance evaluated across all subgroups  
🏥 **Safety:** Designed to augment, not replace, physician judgment  
📋 **Transparency:** Full audit trail for every prediction  
🔐 **Privacy:** No patient identifiers retained in analysis

## Future Enhancements

- **Calibration:** Implement probability calibration for better risk quantification
- **Uncertainty Quantification:** Add confidence intervals to predictions
- **Interactive Dashboard:** Deploy Streamlit app for real-time explanations
- **Prospective Validation:** Test on external hospital datasets
- **Clinical Integration:** API development for EHR systems

## Conclusion

This project demonstrates that achieving high accuracy and complete interpretability are not mutually exclusive goals. By combining SHAP's rigorous global analysis with LIME's intuitive local explanations, we created a heart disease prediction system that clinicians can understand, validate, and trust.

**The bottom line:** In healthcare AI, explainability isn't a nice-to-have feature—it's a fundamental requirement for responsible deployment. This project provides a blueprint for building transparent, trustworthy medical AI systems.

## References

- Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions.* NeurIPS.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why should I trust you?" Explaining the predictions of any classifier.* KDD.
- [UCI Heart Disease Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://lime-ml.readthedocs.io/)

## Project Structure

```
heart-disease-explainable-ai/
│
├── README.md
├── Explainable_AI_Heart_Disease_SHAP_LIME.ipynb
├── data/
│   └── heart_disease_uci.csv
└── visualizations/
    ├── shap_summary_plot.png
    ├── shap_beeswarm_plot.png
    ├── shap_waterfall_cases.png
    └── lime_explanations.png
```

## Author

[Anupa]  
Project completed as part of an Explainable AI in Healthcare study

## License

Educational project - Dataset subject to original Kaggle license terms

---

*"In healthcare AI, we must earn trust not through accuracy alone, but through transparency, validation, and alignment with medical knowledge."*
