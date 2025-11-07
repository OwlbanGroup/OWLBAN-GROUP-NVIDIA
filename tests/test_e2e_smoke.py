import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nvidia_partner_registration_e2e import NVIDIAPartnerRegistration


def test_e2e_registration_smoke(tmp_path):
    """Smoke test: run the E2E registration and expect a successful result.

    This is a lightweight smoke test (not a full integration) that runs the
    `NVIDIAPartnerRegistration.run_e2e_registration()` method and asserts the
    success flag. It writes the report file as a side-effect.
    """
    reg = NVIDIAPartnerRegistration()
    result = reg.run_e2e_registration()

    # Minimal sanity checks
    assert isinstance(result, dict)
    assert result.get("success") is True
    assert result.get("confirmation_number") is not None
