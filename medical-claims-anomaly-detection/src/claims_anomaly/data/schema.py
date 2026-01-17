from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClaimColumns:
    claim_id: str = "claim_id"
    member_id: str = "member_id"
    provider_id: str = "provider_id"
    service_date: str = "service_date"
