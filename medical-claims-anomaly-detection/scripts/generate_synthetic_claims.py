from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class SyntheticConfig:
    seed: int
    n_claims: int
    start_date: str
    end_date: str
    members: Dict
    providers: Dict
    claim_distribution: Dict
    fraud_injection: Dict


def load_config(path: str | Path) -> SyntheticConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return SyntheticConfig(
        seed=int(raw["seed"]),
        n_claims=int(raw["n_claims"]),
        start_date=str(raw["start_date"]),
        end_date=str(raw["end_date"]),
        members=dict(raw["members"]),
        providers=dict(raw["providers"]),
        claim_distribution=dict(raw["claim_distribution"]),
        fraud_injection=dict(raw["fraud_injection"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic medical claims for anomaly detection")
    parser.add_argument("--config", required=True, help="Path to synthetic data config YAML")
    parser.add_argument("--output", required=True, help="Output CSV path")
    return parser.parse_args()


def _choice_from_probs(rng: np.random.Generator, probs: Dict[str, float], size: int) -> np.ndarray:
    keys = list(probs.keys())
    p = np.array([probs[k] for k in keys], dtype=float)
    p = p / p.sum()
    return rng.choice(keys, size=size, p=p)


def _date_range_random(rng: np.random.Generator, start: str, end: str, size: int) -> np.ndarray:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    delta_days = (end_dt - start_dt).days
    offsets = rng.integers(0, max(delta_days, 1) + 1, size=size)
    return (start_dt + pd.to_timedelta(offsets, unit="D")).astype("datetime64[ns]")


def _make_codebook(prefix: str, n: int) -> List[str]:
    return [f"{prefix}{i:04d}" for i in range(1, n + 1)]


def inject_fraud_patterns(
    rng: np.random.Generator,
    df: pd.DataFrame,
    pattern_probs: Dict[str, float],
) -> pd.DataFrame:
    output = df.copy()
    patterns = list(pattern_probs.keys())
    p = np.array([pattern_probs[k] for k in patterns], dtype=float)
    p = p / p.sum()

    fraud_idx = output.index[output["is_fraud_suspected"] == 1].to_numpy()
    if fraud_idx.size == 0:
        return output

    assigned = rng.choice(patterns, size=fraud_idx.size, p=p)

    for idx, pattern in zip(fraud_idx, assigned):
        if pattern == "upcoding":
            output.at[idx, "procedure_code"] = "PROC9999"
            output.at[idx, "allowed_amount"] *= rng.uniform(1.8, 3.2)
            output.at[idx, "paid_amount"] = output.at[idx, "allowed_amount"] * rng.uniform(0.7, 0.9)
        elif pattern == "unbundling":
            output.at[idx, "units"] = int(output.at[idx, "units"] + rng.integers(3, 10))
            output.at[idx, "allowed_amount"] *= rng.uniform(1.3, 2.0)
            output.at[idx, "paid_amount"] = output.at[idx, "allowed_amount"] * rng.uniform(0.65, 0.85)
        elif pattern == "excessive_units":
            output.at[idx, "units"] = int(max(output.at[idx, "units"], 1) * rng.integers(8, 20))
            output.at[idx, "allowed_amount"] *= rng.uniform(1.1, 1.6)
            output.at[idx, "paid_amount"] = output.at[idx, "allowed_amount"] * rng.uniform(0.6, 0.85)
        elif pattern == "phantom_billing":
            output.at[idx, "place_of_service"] = "POS99"
            output.at[idx, "allowed_amount"] *= rng.uniform(0.8, 1.3)
            output.at[idx, "paid_amount"] = output.at[idx, "allowed_amount"] * rng.uniform(0.1, 0.4)
        output.at[idx, "fraud_pattern"] = pattern

    return output


def generate(cfg: SyntheticConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    n_members = int(cfg.members["n_members"])
    n_providers = int(cfg.providers["n_providers"])

    member_ids = np.array([f"M{m:07d}" for m in range(1, n_members + 1)])
    provider_ids = np.array([f"P{p:06d}" for p in range(1, n_providers + 1)])

    member_age = rng.integers(int(cfg.members["age_min"]), int(cfg.members["age_max"]) + 1, size=n_members)
    member_gender = _choice_from_probs(rng, cfg.members["gender_probs"], size=n_members)

    provider_specialty = _choice_from_probs(rng, cfg.providers["specialty_probs"], size=n_providers)

    proc_codes = _make_codebook("PROC", 350)
    diag_codes = _make_codebook("DIAG", 250)
    pos_codes = ["POS11", "POS21", "POS22", "POS23", "POS24", "POS31"]
    bill_types = ["BT01", "BT02", "BT03", "BT04"]

    claim_member = rng.choice(member_ids, size=cfg.n_claims)
    claim_provider = rng.choice(provider_ids, size=cfg.n_claims)
    service_dates = _date_range_random(rng, cfg.start_date, cfg.end_date, size=cfg.n_claims)

    allowed = rng.lognormal(
        mean=float(cfg.claim_distribution["allowed_amount_log_mean"]),
        sigma=float(cfg.claim_distribution["allowed_amount_log_sigma"]),
        size=cfg.n_claims,
    )
    paid_ratio = rng.normal(
        loc=float(cfg.claim_distribution["paid_to_allowed_mean"]),
        scale=float(cfg.claim_distribution["paid_to_allowed_sigma"]),
        size=cfg.n_claims,
    )
    paid_ratio = np.clip(paid_ratio, 0.05, 0.98)
    paid = allowed * paid_ratio

    units = rng.poisson(lam=float(cfg.claim_distribution["units_lambda"]), size=cfg.n_claims) + 1
    days_supply = rng.integers(0, 31, size=cfg.n_claims)

    proc = rng.choice(proc_codes, size=cfg.n_claims)
    diag = rng.choice(diag_codes, size=cfg.n_claims)
    pos = rng.choice(pos_codes, size=cfg.n_claims)
    bt = rng.choice(bill_types, size=cfg.n_claims)

    member_index = pd.Index(member_ids)
    member_age_map = pd.Series(member_age, index=member_index)
    member_gender_map = pd.Series(member_gender, index=member_index)

    provider_index = pd.Index(provider_ids)
    provider_specialty_map = pd.Series(provider_specialty, index=provider_index)

    df = pd.DataFrame(
        {
            "claim_id": [f"C{i:010d}" for i in range(1, cfg.n_claims + 1)],
            "member_id": claim_member,
            "provider_id": claim_provider,
            "service_date": pd.to_datetime(service_dates).dt.strftime("%Y-%m-%d"),
            "procedure_code": proc,
            "diagnosis_code": diag,
            "place_of_service": pos,
            "billing_type": bt,
            "allowed_amount": allowed.round(2),
            "paid_amount": paid.round(2),
            "units": units.astype(int),
            "days_supply": days_supply.astype(int),
            "member_age": member_age_map.loc[claim_member].values.astype(int),
            "member_gender": member_gender_map.loc[claim_member].values,
            "provider_specialty": provider_specialty_map.loc[claim_provider].values,
        }
    )

    fraud_rate = float(cfg.fraud_injection.get("fraud_rate", 0.01))
    is_fraud = (rng.random(cfg.n_claims) < fraud_rate).astype(int)
    df["is_fraud_suspected"] = is_fraud
    df["fraud_pattern"] = "none"

    df = inject_fraud_patterns(rng, df, dict(cfg.fraud_injection.get("patterns", {})))

    return df


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    df = generate(cfg)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
