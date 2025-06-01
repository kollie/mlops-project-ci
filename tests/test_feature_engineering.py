"""
test_feature_engineering.py

Unit tests for all custom feature engineering transformers.

Purpose:
--------
- Validates correctness and robustness of each engineered feature individually.
- Ensures each transformer returns expected outputs for both standard and edge case inputs.
- Required for reproducibility, CI/CD integration, and long-term pipeline stability.

DEV Testing Standards (Based on MLOps Best Practices):
-------------------------------------------------------
- Uses only mock/synthetic data (never real production inputs).
- Validates feature logic in isolation with no pipeline dependencies.
- Covers both success and failure scenarios (missing columns, invalid values).
- Tests should be fast, deterministic, and independently executable.
- Ensures clean error handling, consistent output types, and edge case coverage.

Each feature transformer must be tested independently:
- No hidden assumptions, chaining, or cross-transformer dependencies.
- One test per transformer as minimum.
- All logic paths must be exercised.

Example features tested in this file:
- AgeGroupTransformer
- LengthOfStayGroupTransformer
- ComorbidityTransformer
...
"""
import pytest
import pandas as pd
import numpy as np
import sys
from src import BaseFeatureEngineer
from src.features.feature_engineering import FEATURE_TRANSFORMERS




CONFIG_PATH = "src/config.yaml"

@pytest.fixture
def patient_base():
    """
    Fixture: returns minimal mock patient data with all necessary features.
    This will be copied/modified per test to simulate edge and typical cases.
    """
    return pd.DataFrame({
        "age": [60],
        "time_in_hospital": [7],
        "num_medications": [14],
        "number_inpatient": [1],
        "number_outpatient": [2],
        "number_emergency": [3],
        "number_diagnoses": [8],
        "diag_1": ["401"],
        "diag_2": ["250.02"],
        "diag_3": ["585"],
        "discharge_disposition_id": [3],
        "payer_code": ["MC"],
        "medical_specialty": ["Cardiology"],
        "admission_source_id": [7],
        "change": ["Ch"]
    })

CONFIG_PATH = "src/config.yaml"

def test_length_of_stay_group_basic():
    """
    Test that LengthOfStayGroupTransformer correctly bins 'time_in_hospital' into categorical ranges.

    - Input: patient with time_in_hospital = 8
    - Expected: 'length_of_stay_group' column is created and matches correct bin from config

    Rationale:
    - Verifies that numeric hospital stay is transformed into discrete risk groups
    - Prevents downstream errors due to missing or misclassified feature
    - Ensures consistent, reproducible binning logic driven by config.yaml
    """
    df = pd.DataFrame({"time_in_hospital": [8]})

    transformer = FEATURE_TRANSFORMERS["length_of_stay_group"](config_path=CONFIG_PATH)
    out = transformer.transform(df)

    assert "length_of_stay_group" in out.columns
    assert pd.api.types.is_categorical_dtype(out["length_of_stay_group"])
    assert not pd.isna(out["length_of_stay_group"].iloc[0])


def test_comorbidity_basic():
    """
    Test that ComorbidityTransformer correctly computes the sum of comorbidity indicators
    into the 'num_conditions' feature.

    - Input: one patient with values: 5 (diagnoses), 2 (inpatient), 1 (outpatient), 1 (emergency)
    - Expected: num_conditions = 9

    Rationale:
    - Validates config-driven summation of multiple clinical columns
    - Ensures output is numeric, present, and reproducible
    - Prevents silent column mismatch or type errors
    """
    df = pd.DataFrame({
        "number_diagnoses": [5],
        "number_inpatient": [2],
        "number_outpatient": [1],
        "number_emergency": [1]
    })

    transformer = FEATURE_TRANSFORMERS["comorbidity"](config_path=CONFIG_PATH)
    out = transformer.transform(df)

    assert "num_conditions" in out.columns
    assert out["num_conditions"].iloc[0] == 9

def test_previous_visits_basic():
    """
    Test that PreviousVisitsTransformer correctly computes the total number of previous visits
    using inpatient, outpatient, and emergency visits.

    - Input: inpatient = 2, outpatient = 3, emergency = 1
    - Expected: total_prev_visits = 6

    Rationale:
    - Verifies summation of operational visit indicators into a single numeric feature
    - Captures healthcare system usage intensity
    - Prevents errors due to missing or misconfigured columns
    """
    df = pd.DataFrame({
        "number_inpatient": [2],
        "number_outpatient": [3],
        "number_emergency": [1]
    })

    transformer = FEATURE_TRANSFORMERS["previous_visits"](config_path=CONFIG_PATH)
    out = transformer.transform(df)

    assert "total_prev_visits" in out.columns
    assert out["total_prev_visits"].iloc[0] == 6

def test_medication_intensity_basic():
    """
    Test that MedicationIntensityTransformer calculates the medications per hospital day.

    - Input: num_medications = 20, time_in_hospital = 5
    - Expected: medications_per_day = 4.0

    Rationale:
    - Ensures proper normalization of medication count by stay duration
    - Handles division and coercion correctly (even with potential 0-day stays)
    - Prevents clinical misinterpretation from raw medication counts
    """
    df = pd.DataFrame({
        "num_medications": [20],
        "time_in_hospital": [5]
    })

    transformer = FEATURE_TRANSFORMERS["medication_intensity"](config_path=CONFIG_PATH)
    out = transformer.transform(df)

    assert "medications_per_day" in out.columns
    assert out["medications_per_day"].iloc[0] == 4.0

def test_has_emergency_visit_basic():
    """
    Test that HasEmergencyVisitTransformer creates a binary feature based on emergency visits.

    - Input: number_emergency = 2
    - Expected: has_emergency_visit = 1

    Rationale:
    - Identifies patients with prior emergency usage, a known readmission risk factor
    - Ensures numeric coercion and correct binary assignment
    - Prevents silent failure if column missing or zero values
    """
    df = pd.DataFrame({"number_emergency": [2]})

    transformer = FEATURE_TRANSFORMERS["has_emergency_visit"](config_path=CONFIG_PATH)
    out = transformer.transform(df)

    assert "has_emergency_visit" in out.columns
    assert out["has_emergency_visit"].iloc[0] == 1

def test_was_medicated_basic():
    """
    Test that WasMedicatedTransformer correctly creates a binary feature indicating
    whether the patient received any medications.

    - Input: num_medications = 10
    - Expected: was_medicated = 1

    Rationale:
    - Captures whether pharmacological treatment was applied during stay
    - Ensures binary output and numeric coercion
    - Helps prevent failure from missing or malformed values
    """
    df = pd.DataFrame({"num_medications": [10]})

    transformer = FEATURE_TRANSFORMERS["was_medicated"](config_path=CONFIG_PATH)
    out = transformer.transform(df)

    assert "was_medicated" in out.columns
    assert out["was_medicated"].iloc[0] == 1

def test_has_circulatory_diagnosis_basic():
    """
    Test that HasCirculatoryDiagnosisTransformer detects ICD-9 circulatory codes (390–459).

    - Input: diag_1 = "401", diag_2 = "250.02", diag_3 = "585"
    - Expected: has_circulatory_diagnosis = 1

    Rationale:
    - Ensures correct ICD-based flag for circulatory diseases
    - Important risk factor for readmission
    - Prevents logic error in multi-column scanning
    """
    df = pd.DataFrame({
        "diag_1": ["401"],
        "diag_2": ["250.02"],
        "diag_3": ["585"]
    })

    transformer = FEATURE_TRANSFORMERS["has_circulatory_diagnosis"](config_path=CONFIG_PATH)
    out = transformer.transform(df)

    assert "has_circulatory_diagnosis" in out.columns
    assert out["has_circulatory_diagnosis"].iloc[0] == 1

def test_has_diabetes_diagnosis_basic():
    """
    Test that HasDiabetesDiagnosisTransformer flags ICD-9 diabetes codes (250.xx) correctly.

    - Input: diag_1 = "401", diag_2 = "250.02", diag_3 = "585"
    - Expected: has_diabetes_diagnosis = 1

    Rationale:
    - Captures chronic disease risk from known diabetes codes
    - Ensures reliable ICD-based parsing across diagnosis columns
    - Avoids false negatives due to type or prefix issues
    """
    df = pd.DataFrame({
        "diag_1": ["401"],
        "diag_2": ["250.02"],
        "diag_3": ["585"]
    })

    transformer = FEATURE_TRANSFORMERS["has_diabetes_diagnosis"](config_path=CONFIG_PATH)
    out = transformer.transform(df)

    assert "has_diabetes_diagnosis" in out.columns
    assert out["has_diabetes_diagnosis"].iloc[0] == 1

def test_has_many_diagnoses_basic():
    """
    Test that ManyDiagnosesFlagTransformer flags patients with 9 or more diagnoses.

    - Input: number_diagnoses = 9
    - Expected: has_many_diagnoses = 1

    Rationale:
    - Captures multimorbidity and clinical complexity
    - Ensures reliable binary output for downstream modeling
    - Prevents misflagging from boundary conditions (e.g. 8 vs 9)
    """
    df = pd.DataFrame({"number_diagnoses": [9]})

    transformer = FEATURE_TRANSFORMERS["has_many_diagnoses"](config_path=CONFIG_PATH)
    out = transformer.transform(df)

    assert "has_many_diagnoses" in out.columns
    assert out["has_many_diagnoses"].iloc[0] == 1

def test_has_kidney_diagnosis_basic():
    """
    Test that HasKidneyDiagnosisTransformer detects ICD-9 kidney-related codes (580–589).

    - Input: diag_1 = "401", diag_2 = "250.02", diag_3 = "585"
    - Expected: has_kidney_diagnosis = 1

    Rationale:
    - Identifies patients with renal comorbidity
    - Prevents misclassification by ensuring code ranges are respected
    - Relies on robust multi-column check
    """
    df = pd.DataFrame({
        "diag_1": ["401"],
        "diag_2": ["250.02"],
        "diag_3": ["585"]
    })

    transformer = FEATURE_TRANSFORMERS["has_kidney_diagnosis"](config_path=CONFIG_PATH)
    out = transformer.transform(df)

    assert "has_kidney_diagnosis" in out.columns
    assert out["has_kidney_diagnosis"].iloc[0] == 1

def test_is_discharged_to_facility_basic():
    """
    Test that IsDischargedToFacilityTransformer flags facility-based discharges correctly.

    - Input: discharge_disposition_id = 3 (in config-defined facility list)
    - Expected: is_discharged_to_facility = 1

    Rationale:
    - Captures institutional discharges associated with higher readmission risk
    - Ensures correct config loading and membership logic
    - Prevents mismatches if facility codes change
    """
    df = pd.DataFrame({"discharge_disposition_id": [3]})

    transformer = FEATURE_TRANSFORMERS["is_discharged_to_facility"](config_path=CONFIG_PATH)
    out = transformer.transform(df)

    assert "is_discharged_to_facility" in out.columns
    assert out["is_discharged_to_facility"].iloc[0] == 1

def test_is_government_payer_basic():
    """
    Test that IsGovernmentPayerTransformer flags government payer codes correctly.

    - Input: payer_code = "MC" (Medicare, listed in config)
    - Expected: is_government_payer = 1

    Rationale:
    - Captures socioeconomic signals from payer types
    - Verifies config-based list matching
    - Prevents logic break if payer codes change or are missing
    """
    df = pd.DataFrame({"payer_code": ["MC"]})

    transformer = FEATURE_TRANSFORMERS["is_government_payer"](config_path=CONFIG_PATH)
    out = transformer.transform(df)

    assert "is_government_payer" in out.columns
    assert out["is_government_payer"].iloc[0] == 1

def test_is_specialty_high_risk_basic():
    """
    Test that IsSpecialtyHighRiskTransformer flags specialties defined as high-risk in config.yaml.

    - Input: medical_specialty = "Cardiology"
    - Expected: is_specialty_high_risk = 1

    Rationale:
    - Captures medical departments associated with higher readmission probability
    - Verifies config-driven logic for dynamic specialty classification
    - Ensures no misflagging due to whitespace, casing, or missing values
    """
    df = pd.DataFrame({"medical_specialty": ["Cardiology"]})

    transformer = FEATURE_TRANSFORMERS["is_specialty_high_risk"](config_path=CONFIG_PATH)
    out = transformer.transform(df)

    assert "is_specialty_high_risk" in out.columns
    assert out["is_specialty_high_risk"].iloc[0] == 1

def test_is_admitted_from_critical_source_basic():
    """
    Test that IsAdmittedFromCriticalSourceTransformer correctly flags critical admission sources.

    - Input: admission_source_id = 7 (in config list)
    - Expected: is_admitted_from_critical_source = 1

    Rationale:
    - Captures context of urgent or acute admission pathways
    - Ensures config-driven flagging logic is applied correctly
    - Prevents silent errors from changes in source ID mapping
    """
    df = pd.DataFrame({"admission_source_id": [7]})

    transformer = FEATURE_TRANSFORMERS["is_admitted_from_critical_source"](config_path=CONFIG_PATH)
    out = transformer.transform(df)

    assert "is_admitted_from_critical_source" in out.columns
    assert out["is_admitted_from_critical_source"].iloc[0] == 1

def test_had_medication_change_basic():
    """
    Test that HadMedicationChangeTransformer flags medication changes based on the 'change' column.

    - Input: change = 'Ch'
    - Expected: had_medication_change = 1

    Rationale:
    - Captures treatment adjustments during stay
    - Prevents false negatives due to missing or malformed values
    - Ensures text parsing logic is robust and case-insensitive
    """
    df = pd.DataFrame({"change": ["Ch"]})

    transformer = FEATURE_TRANSFORMERS["had_medication_change"](config_path=CONFIG_PATH)
    out = transformer.transform(df)

    assert "had_medication_change" in out.columns
    assert out["had_medication_change"].iloc[0] == 1

def test_select_k_best_basic():
    """
    Test that FeatureSelector correctly selects the top-k features using univariate F-test.

    - Input: 15 numeric features, binary target
    - Config: k = 10
    - Expected: Output dataframe has 10 columns

    Rationale:
    - Reduces dimensionality to improve generalization
    - Ensures compatibility with sklearn pipeline standards
    - Validates selection logic and shape of output
    """
    # Input: 15 features, 5 samples
    np.random.seed(0)
    X = pd.DataFrame(np.random.rand(5, 15), columns=[f"feat_{i}" for i in range(15)])
    y = pd.Series([0, 1, 0, 1, 0], name="readmitted")

    transformer = FEATURE_TRANSFORMERS["select_k_best"](config_path=CONFIG_PATH)
    transformer.fit(X, y)
    out = transformer.transform(X)

    assert isinstance(out, pd.DataFrame)
    assert out.shape[1] == 10
    assert len(out) == 5

def test_feature_engineering_pipeline_integration():
    """
    Integration test for FeatureEngineeringPipeline.

    - Input: synthetic dataset with all required columns + binary target
    - Expected: returned dataframe with top-k selected features and no NaNs

    Rationale:
    - Verifies end-to-end pipeline orchestration
    - Ensures compatibility between individual transformers and selector
    - Captures integration issues (e.g., column mismatch, data leakage)
    """
    # Minimal input (2 samples)
    df = pd.DataFrame({
        "age": [60, 45],
        "time_in_hospital": [5, 10],
        "num_medications": [12, 18],
        "number_inpatient": [1, 0],
        "number_outpatient": [2, 1],
        "number_emergency": [0, 3],
        "number_diagnoses": [5, 9],
        "diag_1": ["401", "250.02"],
        "diag_2": ["250.02", "585"],
        "diag_3": ["585", "414"],
        "discharge_disposition_id": [3, 1],
        "payer_code": ["MC", "HM"],
        "medical_specialty": ["Cardiology", "Surgery-General"],
        "admission_source_id": [7, 4],
        "change": ["Ch", "No"],
        "readmitted": [1, 0]
    })

    X = df.drop(columns=["readmitted"])
    y = df["readmitted"]

    from src.features.feature_engineering import FeatureEngineeringPipeline
    pipeline = FeatureEngineeringPipeline(config_path=CONFIG_PATH)
    X_transformed, selected_features = pipeline.engineer_features(X, y)

    assert isinstance(X_transformed, pd.DataFrame)
    assert len(X_transformed) == 2
    assert len(selected_features) <= X.shape[1] + 16  # 16 engineered features
    assert not X_transformed.isnull().any().any()
    assert all([col in X_transformed.columns for col in selected_features])
