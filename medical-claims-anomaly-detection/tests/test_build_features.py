import pandas as pd

from claims_anomaly.features.build_features import build_features


def test_build_features_runs() -> None:
    df = pd.DataFrame(
        {
            "claim_id": ["C1", "C2"],
            "member_id": ["M1", "M1"],
            "provider_id": ["P1", "P2"],
            "service_date": ["2024-01-01", "2024-01-02"],
            "procedure_code": ["PROC0001", "PROC0002"],
            "diagnosis_code": ["DIAG0001", "DIAG0002"],
            "place_of_service": ["POS11", "POS21"],
            "billing_type": ["BT01", "BT02"],
            "allowed_amount": [100.0, 200.0],
            "paid_amount": [80.0, 150.0],
            "units": [1, 2],
            "days_supply": [0, 10],
            "member_age": [34, 34],
            "member_gender": ["F", "F"],
            "is_fraud_suspected": [0, 1],
        }
    )

    features_cfg = {
        "numeric": ["allowed_amount", "paid_amount", "units", "days_supply", "member_age"],
        "categorical": ["procedure_code", "diagnosis_code", "place_of_service", "billing_type", "member_gender"],
        "datetime": ["service_date"],
        "aggregations": {
            "provider_window_days": 30,
            "member_window_days": 30,
            "include_provider_aggregates": True,
            "include_member_aggregates": True,
        },
    }

    out = build_features(df, features_cfg)
    assert out.feature_matrix.shape[0] == 2
