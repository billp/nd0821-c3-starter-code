# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This is a Random Forest Classifier trained to predict whether a person's annual income exceeds $50K based on census demographic data. The model uses scikit-learn's `RandomForestClassifier` with 100 estimators and `random_state=42` for reproducibility. Categorical features are one-hot encoded and the target label is binarized.

## Intended Use

This model is intended for educational and demonstration purposes as part of a machine learning DevOps course. It should not be used for real-world income prediction or decision-making about individuals.

## Training Data

The training data comes from the UCI Census Income dataset (also known as the "Adult" dataset), extracted from the 1994 Census Bureau database. The full dataset contains 32,561 rows and 14 features (6 continuous, 8 categorical) with a binary target variable (salary: <=50K or >50K). An 80/20 train-test split was used, resulting in approximately 26,048 training samples.

## Evaluation Data

The evaluation data is the 20% holdout test set from the same Census Income dataset, containing approximately 6,513 samples. The same preprocessing pipeline (one-hot encoding, label binarization) fitted on the training data was applied to the test data.

## Metrics

The model was evaluated using precision, recall, and F1 score (F-beta with beta=1).

Overall performance on the test set:
- **Precision**: 0.7391
- **Recall**: 0.6384
- **F1 (F-beta)**: 0.6851

Slice performance analysis was conducted across all 8 categorical features. Notable findings include performance variation across demographic groups (see `slice_output.txt` for full details).

## Ethical Considerations

The model uses sensitive demographic attributes including race, sex, and native country as input features. Predictions may reflect and perpetuate historical biases present in the 1994 census data. The slice performance analysis reveals disparities across demographic groups -- for example, recall varies across racial groups and between sexes. This model should not be used for any decisions that affect individuals.

## Caveats and Recommendations

- The model was trained on data from 1994, which may not reflect current income distributions or demographic patterns.
- Performance varies significantly across data slices, particularly for underrepresented groups with small sample sizes.
- The model has higher precision than recall, meaning it is more conservative in predicting high income (>50K).
- For any real-world application, a thorough fairness audit and more recent data would be required.
