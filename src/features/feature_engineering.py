
"""
feature_engineering_refactored.py

Advanced feature engineering using scikit-learn compatible transformers.
Structured for integration in MLOps pipelines.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
import logging
import yaml
from typing import List, Tuple, Dict, Any
from pathlib import Path


class BaseFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, config_path="src/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()

    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.error(f"Failed to load config from {self.config_path}: {e}")
            raise


    def _setup_logging(self):
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
              level=self.config['logging']['level'],
              format=self.config['logging']['format'],
              filename=self.config['logging']['file']
            )
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y=None):
        return self


class AgeGroupTransformer(BaseFeatureEngineer):

    """
    Adds an 'age_group' column: binned version of the continuous 'age' feature.

    Clinical motivation:
    - Grouping patient age into ranges captures non-linear effects.
    - Improves model interpretability and allows categorical treatment of age bands.

    Configuration:
    - Uses 'age_bins' and 'age_labels' defined in config.yaml under 'features'.

    Output:
    - Adds a new column 'age_group' to the dataset as a categorical feature.

    Usage:
        pipeline = Pipeline([
            ('AgeGroup', AgeGroupTransformer(config_path='src/config.yaml')),
            ...
        ])
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        super().__init__(config_path)

    def fit(self, X: pd.DataFrame, y=None) -> 'AgeGroupTransformer':
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if 'age' not in X.columns:
            self.logger.error("'age' column is missing from input data.")
            raise ValueError("'age' column is required to create 'age_group'.")

        if 'age_group' in X.columns:
            self.logger.warning("'age_group' column already exists and will be overwritten.")

        X['age'] = pd.to_numeric(X['age'], errors='coerce')

        X['age_group'] = pd.cut(
            X['age'],
            bins=self.config['features']['age_bins'],
            labels=self.config['features']['age_labels']
        )

        self.logger.info("Created 'age_group' feature using age_bins and labels from config.yaml.")
        return X

class LengthOfStayGroupTransformer(BaseFeatureEngineer):

    """
    Adds a 'length_of_stay_group' column: categorized version of 'time_in_hospital'.

    Clinical motivation:
    - The duration of hospital stay is a strong indicator of patient severity and resource use.
    - Grouping this value helps capture patterns while reducing model variance and overfitting.

    Configuration:
    - Uses 'los_bins' and 'los_labels' defined in config.yaml under 'features'.

    Output:
    - Adds a new column 'length_of_stay_group' to the dataset as a categorical feature.

    Usage:
        pipeline = Pipeline([
            ('LengthOfStayGroup', LengthOfStayGroupTransformer(config_path='src/config.yaml')),
            ...
        ])
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        super().__init__(config_path)

    def fit(self, X: pd.DataFrame, y=None) -> 'LengthOfStayGroupTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if 'time_in_hospital' not in X.columns:
            self.logger.error("'time_in_hospital' column is missing from input data.")
            raise ValueError("'time_in_hospital' column is required to create 'length_of_stay_group'.")

        if 'length_of_stay_group' in X.columns:
            self.logger.warning("'length_of_stay_group' column already exists and will be overwritten.")

        X['time_in_hospital'] = pd.to_numeric(X['time_in_hospital'], errors='coerce')

        X['length_of_stay_group'] = pd.cut(
            X['time_in_hospital'],
            bins=self.config['features']['los_bins'],
            labels=self.config['features']['los_labels']
        )

        self.logger.info("Created 'length_of_stay_group' feature using los_bins and los_labels from config.yaml.")
        return X
    
class ComorbidityTransformer(BaseFeatureEngineer):
    
    """
    Adds a 'num_conditions' feature that combines multiple indicators of comorbidity:
    - number_diagnoses
    - number_inpatient
    - number_outpatient
    - number_emergency (read from config.yaml)

    Clinical motivation:
    - Aggregates comorbidity and prior healthcare usage into a single interpretable metric.
    - Higher values suggest complex patients with increased risk of readmission.

    Output:
    - Adds a new column 'num_conditions' to the dataset.

    Usage:
        pipeline = Pipeline([
            ('Comorbidity', ComorbidityTransformer(config_path='src/config.yaml')),
            ...
        ])
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        super().__init__(config_path)

    def fit(self, X: pd.DataFrame, y=None) -> 'ComorbidityTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        required_columns = self.config['features']['comorbidity_columns']

        missing = [col for col in required_columns if col not in X.columns]
        if missing:
            self.logger.error(f"Missing columns for ComorbidityTransformer: {missing}")
            raise ValueError(f"The following required columns are missing: {missing}")

        if 'num_conditions' in X.columns:
            self.logger.warning("'num_conditions' column already exists and will be overwritten.")

        X['num_conditions'] = X[required_columns].apply(pd.to_numeric, errors='coerce').sum(axis=1)

        self.logger.info("Created 'num_conditions' feature by combining columns from config.yaml.")
        return X

class PreviousVisitsTransformer(BaseFeatureEngineer):
    """
    Adds a 'total_prev_visits' feature combining:
    - number_inpatient
    - number_outpatient
    - number_emergency (read from config.yaml)

    Clinical motivation:
    - Captures the total frequency of prior hospital interactions.
    - Patients who frequently visit emergency, outpatient or inpatient services are often unstable, poorly managed, or chronically ill.
    - This variable serves as a proxy for healthcare utilization burden.

    Why it complements 'num_conditions':
    - 'num_conditions' includes number_diagnoses, introducing a clinical component.
    - 'total_prev_visits' isolates the operational side: how often the patient actually uses the system, regardless of diagnosis count.
    - Thus, even if correlated, both capture **different aspects of risk**: clinical complexity vs. system dependency.
    - Retaining both enables models to learn richer interactions (e.g., frequent visits with few diagnoses might imply misdiagnosed or poorly managed cases).

    Output:
    - Adds a new column 'total_prev_visits' to the dataset.

    Usage:
        pipeline = Pipeline([
            ('PreviousVisits', PreviousVisitsTransformer(config_path='src/config.yaml')),
            ...
        ])
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        super().__init__(config_path)

    def fit(self, X: pd.DataFrame, y=None) -> 'PreviousVisitsTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        required_columns = self.config['features']['previous_visits_columns']

        missing = [col for col in required_columns if col not in X.columns]
        if missing:
            self.logger.error(f"Missing columns for PreviousVisitsTransformer: {missing}")
            raise ValueError(f"The following required columns are missing: {missing}")

        if 'total_prev_visits' in X.columns:
            self.logger.warning("'total_prev_visits' column already exists and will be overwritten.")

        X['total_prev_visits'] = X[required_columns].apply(pd.to_numeric, errors='coerce').sum(axis=1)

        self.logger.info("Created 'total_prev_visits' feature using columns from config.yaml.")
        return X

class MedicationIntensityTransformer(BaseFeatureEngineer):
    """
    Adds a 'medications_per_day' feature:
    - Calculated as: num_medications / time_in_hospital

    Clinical motivation:
    - Captures the intensity of pharmacological treatment per hospital day.
    - High values may indicate acute complexity or polypharmacy.
    - Low values could indicate treatment abandonment or insufficient care.
    
    Why it's useful:
    - Normalizes num_medications relative to hospital stay length.
    - Adds context: 15 medications in 2 days ≠ 15 medications in 10 days.
    - Helps identify outlier care patterns and treatment burden.

    Precautions:
    - Avoids division by zero by applying a floor of 1 to time_in_hospital.

    Output:
    - Adds a new column 'medications_per_day' to the dataset.

    Usage:
        pipeline = Pipeline([
            ('MedicationIntensity', MedicationIntensityTransformer(config_path='src/config.yaml')),
            ...
        ])
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        super().__init__(config_path)

    def fit(self, X: pd.DataFrame, y=None) -> 'MedicationIntensityTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if 'num_medications' not in X.columns or 'time_in_hospital' not in X.columns:
            missing = [col for col in ['num_medications', 'time_in_hospital'] if col not in X.columns]
            self.logger.error(f"Missing columns for MedicationIntensityTransformer: {missing}")
            raise ValueError(f"The following required columns are missing: {missing}")

        if 'medications_per_day' in X.columns:
            self.logger.warning("'medications_per_day' column already exists and will be overwritten.")

        meds = pd.to_numeric(X['num_medications'], errors='coerce')
        days = pd.to_numeric(X['time_in_hospital'], errors='coerce').replace(0, 1)  # avoid division by zero

        X['medications_per_day'] = meds / days

        self.logger.info("Created 'medications_per_day' feature as num_medications / time_in_hospital.")
        return X

class HasEmergencyVisitTransformer(BaseFeatureEngineer):
    """
    Adds a binary feature 'has_emergency_visit' indicating whether the patient
    had at least one emergency room visit before this hospital admission.

    Clinical motivation:
    - Emergency visits reflect clinical instability or crisis episodes.
    - Patients with prior emergency usage are at higher risk of readmission.

    Output:
    - Adds column 'has_emergency_visit': 1 if number_emergency > 0, else 0.
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        super().__init__(config_path)

    def fit(self, X: pd.DataFrame, y=None) -> 'HasEmergencyVisitTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if 'number_emergency' not in X.columns:
            self.logger.error("Missing column 'number_emergency' for HasEmergencyVisitTransformer.")
            raise ValueError("Required column 'number_emergency' is missing.")

        if 'has_emergency_visit' in X.columns:
            self.logger.warning("'has_emergency_visit' column already exists and will be overwritten.")

        X['has_emergency_visit'] = (pd.to_numeric(X['number_emergency'], errors='coerce') > 0).astype(int)

        self.logger.info("Created 'has_emergency_visit' feature based on emergency visit count.")
        return X

class WasMedicatedTransformer(BaseFeatureEngineer):
    """
    Adds a binary feature 'was_medicated' indicating whether the patient
    received any medications during their hospital stay.

    Clinical motivation:
    - Active medication during stay implies treatment complexity.
    - Patients with pharmacological interventions may have higher risk of readmission.

    Output:
    - Adds column 'was_medicated': 1 if num_medications > 0, else 0.
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        super().__init__(config_path)

    def fit(self, X: pd.DataFrame, y=None) -> 'WasMedicatedTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if 'num_medications' not in X.columns:
            self.logger.error("Missing column 'num_medications' for WasMedicatedTransformer.")
            raise ValueError("Required column 'num_medications' is missing.")

        if 'was_medicated' in X.columns:
            self.logger.warning("'was_medicated' column already exists and will be overwritten.")

        X['was_medicated'] = (pd.to_numeric(X['num_medications'], errors='coerce') > 0).astype(int)

        self.logger.info("Created 'was_medicated' feature based on num_medications.")
        return X

class HasCirculatoryDiagnosisTransformer(BaseFeatureEngineer):
    """
    Adds 'has_circulatory_diagnosis': 1 if any of diag_1, diag_2 or diag_3 is a circulatory system condition (ICD-9 390–459).

    Clinical motivation:
    - Circulatory diagnoses (heart failure, ischemia, hypertension) are strong predictors of hospital readmission.
    - Indicates chronic disease burden and cardiovascular instability.

    Output:
    - Adds column 'has_circulatory_diagnosis': 1 if any diagnosis is between 390 and 459, else 0.
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        super().__init__(config_path)

    def fit(self, X: pd.DataFrame, y=None) -> 'HasCirculatoryDiagnosisTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        diag_cols = ['diag_1', 'diag_2', 'diag_3']

        for col in diag_cols:
            if col not in X.columns:
                self.logger.error(f"Missing column '{col}' for circulatory diagnosis feature.")
                raise ValueError(f"Required column '{col}' is missing.")

        def is_circulatory(code):
            try:
                numeric = float(str(code).replace('E', '').replace('V', ''))
                return 390 <= numeric <= 459
            except ValueError:
                return False

        X['has_circulatory_diagnosis'] = X[diag_cols].apply(
            lambda row: any(is_circulatory(code) for code in row), axis=1
        ).astype(int)

        self.logger.info("Created 'has_circulatory_diagnosis' feature.")
        return X

class HasDiabetesDiagnosisTransformer(BaseFeatureEngineer):
    """
    Adds 'has_diabetes_diagnosis': 1 if any of diag_1, diag_2 or diag_3 is a diabetes diagnosis (ICD-9 250.xx).

    Clinical motivation:
    - Diabetes mellitus is a chronic disease with high risk of readmission.
    - Its presence increases the complexity of management and likelihood of acute events.

    Output:
    - Adds column 'has_diabetes_diagnosis': 1 if any diagnosis is in the ICD-9 range 250.00–250.99, else 0.
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        super().__init__(config_path)

    def fit(self, X: pd.DataFrame, y=None) -> 'HasDiabetesDiagnosisTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        diag_cols = ['diag_1', 'diag_2', 'diag_3']

        for col in diag_cols:
            if col not in X.columns:
                self.logger.error(f"Missing column '{col}' for diabetes diagnosis feature.")
                raise ValueError(f"Required column '{col}' is missing.")

        def is_diabetes(code):
            try:
                code = str(code)
                if code.startswith("250"):
                    numeric = float(code)
                    return 250.0 <= numeric < 251.0
                return False
            except ValueError:
                return False

        X['has_diabetes_diagnosis'] = X[diag_cols].apply(
            lambda row: any(is_diabetes(code) for code in row), axis=1
        ).astype(int)

        self.logger.info("Created 'has_diabetes_diagnosis' feature.")
        return X

class ManyDiagnosesFlagTransformer(BaseFeatureEngineer):
    """
    Adds 'has_many_diagnoses': 1 if number_diagnoses >= 9.

    Clinical motivation:
    - A high number of diagnoses often indicates complex, multimorbid patients.
    - These patients are more likely to be readmitted due to multiple care needs.

    Output:
    - Adds column 'has_many_diagnoses': 1 if number_diagnoses >= 9, else 0.
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        super().__init__(config_path)

    def fit(self, X: pd.DataFrame, y=None) -> 'ManyDiagnosesFlagTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if 'number_diagnoses' not in X.columns:
            self.logger.error("Missing column 'number_diagnoses' for ManyDiagnosesFlagTransformer.")
            raise ValueError("Required column 'number_diagnoses' is missing.")

        if 'has_many_diagnoses' in X.columns:
            self.logger.warning("'has_many_diagnoses' already exists and will be overwritten.")

        X['has_many_diagnoses'] = (pd.to_numeric(X['number_diagnoses'], errors='coerce') >= 9).astype(int)

        self.logger.info("Created 'has_many_diagnoses' feature.")
        return X

class HasKidneyDiagnosisTransformer(BaseFeatureEngineer):
    """
    Adds 'has_kidney_diagnosis': 1 if any of diag_1, diag_2 or diag_3 is a kidney condition (ICD-9 580–589).

    Clinical motivation:
    - Kidney diseases are common comorbidities that increase risk of readmission.
    - They impact fluid/electrolyte balance and require chronic management.

    Output:
    - Adds column 'has_kidney_diagnosis': 1 if any diagnosis is between 580 and 589, else 0.
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        super().__init__(config_path)

    def fit(self, X: pd.DataFrame, y=None) -> 'HasKidneyDiagnosisTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        diag_cols = ['diag_1', 'diag_2', 'diag_3']

        for col in diag_cols:
            if col not in X.columns:
                self.logger.error(f"Missing column '{col}' for kidney diagnosis feature.")
                raise ValueError(f"Required column '{col}' is missing.")

        def is_kidney(code):
            try:
                code = str(code)
                if code.startswith('E') or code.startswith('V'):
                    return False
                numeric = float(code)
                return 580 <= numeric <= 589
            except ValueError:
                return False

        X['has_kidney_diagnosis'] = X[diag_cols].apply(
            lambda row: any(is_kidney(code) for code in row), axis=1
        ).astype(int)

        self.logger.info("Created 'has_kidney_diagnosis' feature.")
        return X

class IsDischargedToFacilityTransformer(BaseFeatureEngineer):
    """
    Adds 'is_discharged_to_facility': 1 if discharge_disposition_id corresponds to a care facility.

    Clinical motivation:
    - Discharge to facilities (rehab, SNF, hospice, psychiatric hospitals) reflects higher illness burden.
    - These patients are at greater risk of readmission than those discharged home.

    Output:
    - Adds column 'is_discharged_to_facility': 1 if discharge_disposition_id is in facility codes from config.yaml, else 0.
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        super().__init__(config_path)
        self.facility_codes = [int(code) for code in self.config.get("facility_discharge_codes", [])]

    def fit(self, X: pd.DataFrame, y=None) -> 'IsDischargedToFacilityTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if 'discharge_disposition_id' not in X.columns:
            self.logger.error("Missing column 'discharge_disposition_id' for discharge feature.")
            raise ValueError("Required column 'discharge_disposition_id' is missing.")

        X['is_discharged_to_facility'] = X['discharge_disposition_id'].isin(self.facility_codes).astype(int)

        self.logger.info("Created 'is_discharged_to_facility' feature.")
        return X

class IsGovernmentPayerTransformer(BaseFeatureEngineer):
    """
    Adds 'is_government_payer': 1 if payer_code is Medicare, Medicaid, or VA (MC, MD, CM).

    Clinical motivation:
    - Government payers (Medicare, Medicaid, VA) are associated with chronic conditions, socioeconomic vulnerability, and higher readmission risk.
    - These signals are relevant for predictive modeling.

    Output:
    - Adds column 'is_government_payer': 1 if payer_code in government_payer_codes (config.yaml), else 0.
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        super().__init__(config_path)
        self.gov_payer_codes = self.config.get("government_payer_codes", ["MC", "MD", "CM"])

    def fit(self, X: pd.DataFrame, y=None) -> 'IsGovernmentPayerTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if 'payer_code' not in X.columns:
            self.logger.error("Missing column 'payer_code' for insurance feature.")
            raise ValueError("Required column 'payer_code' is missing.")

        X['is_government_payer'] = X['payer_code'].apply(
            lambda code: 1 if code in self.gov_payer_codes else 0
        ).astype(int)

        self.logger.info("Created 'is_government_payer' feature.")
        return X

class IsSpecialtyHighRiskTransformer(BaseFeatureEngineer):
    """
    Adds 'is_specialty_high_risk': 1 if medical_specialty is a high-risk specialty for readmission.

    Clinical motivation:
    - Certain specialties are associated with chronic, unstable, or complex patients.
    - These include Internal Medicine, Cardiology, Nephrology, etc.

    Output:
    - Adds column 'is_specialty_high_risk': 1 if specialty in config high_risk list, else 0.
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        super().__init__(config_path)
        self.high_risk_specialties = self.config.get("high_risk_specialties", [])

    def fit(self, X: pd.DataFrame, y=None) -> 'IsSpecialtyHighRiskTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if 'medical_specialty' not in X.columns:
            self.logger.error("Missing column 'medical_specialty' for specialty feature.")
            raise ValueError("Required column 'medical_specialty' is missing.")

        X['is_specialty_high_risk'] = X['medical_specialty'].fillna("").apply(
            lambda x: 1 if str(x).strip() in self.high_risk_specialties else 0
        ).astype(int)

        self.logger.info("Created 'is_specialty_high_risk' feature.")
        return X

class IsAdmittedFromCriticalSourceTransformer(BaseFeatureEngineer):
    """
    Adds 'is_admitted_from_critical_source': 1 if admission_source_id is in high-risk admission sources.

    Clinical motivation:
    - Admissions from ER, nursing homes, or other hospitals often reflect acute deterioration.
    - These are linked to higher readmission rates.

    Output:
    - Adds column 'is_admitted_from_critical_source': 1 if ID is in critical list from config.yaml.
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        super().__init__(config_path)
        self.critical_sources = self.config.get("critical_admission_sources", [])

    def fit(self, X: pd.DataFrame, y=None) -> 'IsAdmittedFromCriticalSourceTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if 'admission_source_id' not in X.columns:
            self.logger.error("Missing column 'admission_source_id'.")
            raise ValueError("Required column 'admission_source_id' is missing.")

        X['is_admitted_from_critical_source'] = X['admission_source_id'].isin(self.critical_sources).astype(int)

        self.logger.info("Created 'is_admitted_from_critical_source' feature.")
        return X

class HadMedicationChangeTransformer(BaseFeatureEngineer):
    """
    Adds 'had_medication_change': 1 if the patient had a change in medications during the stay.

    Clinical motivation:
    - Medication changes often reflect clinical instability or treatment adjustments.
    - These factors are associated with a higher risk of early readmission.

    Output:
    - Adds column 'had_medication_change': 1 if change == 'Ch', else 0.
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        super().__init__(config_path)

    def fit(self, X: pd.DataFrame, y=None) -> 'HadMedicationChangeTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if 'change' not in X.columns:
            self.logger.error("Missing column 'change' for medication feature.")
            raise ValueError("Required column 'change' is missing.")

        X['had_medication_change'] = X['change'].fillna("").apply(
            lambda x: 1 if str(x).strip().lower().startswith("ch") else 0
        ).astype(int)

        self.logger.info("Created 'had_medication_change' feature.")
        return X


class FeatureSelector(BaseFeatureEngineer):

    """
    Selects the top-k features using univariate statistical tests (ANOVA F-score).

    Motivation:
    - Reduces dimensionality and improves model generalization.
    - Keeps only the most statistically relevant features.

    Configuration:
    - Reads 'n_features_to_select' from config.yaml under 'features'.

    Output:
    - A DataFrame with only the top-k selected features.

    Usage:
        pipeline = Pipeline([
            ('select_k_best', FeatureSelector(config_path='src/config.yaml')),
            ...
        ])
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        super().__init__(config_path)
        self.k = self.config['features']['n_features_to_select']
        self.selector = SelectKBest(score_func=f_classif, k=self.k)
        self.selected_features: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        if X.empty or X.shape[1] == 0:
            self.logger.error("FeatureSelector received an empty feature matrix.")
            raise ValueError("Input X has no features to select from.")

        self.selector.fit(X, y)
        self.selected_features = X.columns[self.selector.get_support()].tolist()
        self.logger.info(f"Selected top {self.k} features: {self.selected_features}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_selected = self.selector.transform(X)
        return pd.DataFrame(X_selected, columns=self.selected_features)

FEATURE_TRANSFORMERS = {
     "age_group": AgeGroupTransformer,
    "length_of_stay_group": LengthOfStayGroupTransformer,
    "comorbidity": ComorbidityTransformer,
    "previous_visits": PreviousVisitsTransformer,
    "medication_intensity": MedicationIntensityTransformer,
    "has_emergency_visit": HasEmergencyVisitTransformer,
    "was_medicated": WasMedicatedTransformer,
    "has_circulatory_diagnosis": HasCirculatoryDiagnosisTransformer,
    "has_diabetes_diagnosis": HasDiabetesDiagnosisTransformer,
    "has_many_diagnoses": ManyDiagnosesFlagTransformer,
    "has_kidney_diagnosis": HasKidneyDiagnosisTransformer,
    "is_discharged_to_facility": IsDischargedToFacilityTransformer,
    "is_government_payer": IsGovernmentPayerTransformer,
    "is_specialty_high_risk": IsSpecialtyHighRiskTransformer,
    "is_admitted_from_critical_source": IsAdmittedFromCriticalSourceTransformer,
    "had_medication_change": HadMedicationChangeTransformer,
    "select_k_best": FeatureSelector
}

class FeatureEngineeringPipeline(BaseFeatureEngineer):

    """
    Pipeline class to orchestrate feature engineering steps:
    - Applies feature creation transformers
    - Applies feature selection using FeatureSelector
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        super().__init__(config_path)

    def engineer_features(self, data: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """
        Execute the full feature engineering pipeline:
        - Apply all feature creation transformers
        - Apply SelectKBest to reduce dimensionality

        Args:
            data (pd.DataFrame): Input dataset
            target (pd.Series): Target variable

        Returns:
            Tuple[pd.DataFrame, List[str]]: Transformed dataset and list of selected features
        """
        try:
            data_with_features = data.copy()

            for name, transformer_class in FEATURE_TRANSFORMERS.items():
                if name != "select_k_best":
                    transformer = transformer_class(config_path=self.config_path)
                    data_with_features = transformer.fit_transform(data_with_features)

            selector = FEATURE_TRANSFORMERS["select_k_best"](config_path=self.config_path)
            X_selected = selector.fit_transform(data_with_features.drop(columns=[target.name]), target)
            selected_features = selector.selected_features

            self.logger.info("Feature engineering pipeline completed successfully.")
            return X_selected, selected_features

        except Exception as e:
            self.logger.error(f"Error in feature engineering pipeline: {str(e)}")
            raise

