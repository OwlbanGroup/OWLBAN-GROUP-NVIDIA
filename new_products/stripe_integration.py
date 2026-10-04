import os
import logging

logger = logging.getLogger(__name__)

# Optional Stripe SDK. Payment features degrade gracefully when the SDK (or
# STRIPE_API_KEY) is unavailable, so the rest of the CombinedSystem can still
# import and run end-to-end. This mirrors the optional-dependency pattern used
# across the codebase (qiskit, docker, tensorrt, cupy, ...).
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    stripe = None
    STRIPE_AVAILABLE = False
    logger.info("stripe package not installed; StripeIntegration running in simulation mode")


class StripeIntegration:
    def __init__(self):
        # Load Stripe API key from environment variable
        self.api_key = os.getenv("STRIPE_API_KEY")
        # STRIPE_ORG_ACCOUNT is the target account (acct_...) an Organization
        # key (sk_org_) acts on, sent via the Stripe-Context header. Only
        # required for org-level keys, not account-level sk_/rk_ keys.
        self.stripe_account = os.getenv("STRIPE_ORG_ACCOUNT")
        self.stripe_available = STRIPE_AVAILABLE and self.api_key is not None
        if not self.api_key:
            # Use test key for development/demo purposes
            self.api_key = "sk_test_dummy_key_for_development"
            logger.warning(
                "Using dummy Stripe API key for development. Set STRIPE_API_KEY "
                "environment variable for production."
            )
        if STRIPE_AVAILABLE:
            stripe.api_key = self.api_key
            # Organization-level keys (sk_org_*) act across accounts in a Stripe
            # Organization and require Stripe-Version + Stripe-Context headers on
            # every request. Account-level keys (sk_live_/rk_live_) do not.
            if isinstance(self.api_key, str) and self.api_key.startswith("sk_org_"):
                self._configure_org_headers()

    def _configure_org_headers(self):
        """Attach the Stripe-Version + Stripe-Context headers required by
        Organization-level API keys (sk_org_*).

        These keys operate across the accounts of a Stripe Organization, so each
        request must name the target account via Stripe-Context and pin an API
        version via Stripe-Version. Account-level keys need neither. Backward
        compatible: account-level sk_/rk_ keys set no extra headers.
        """
        if not self.stripe_account:
            raise RuntimeError(
                "STRIPE_API_KEY is an organization key (sk_org_) but no target "
                "account is configured. Set STRIPE_ORG_ACCOUNT=<acct_...> (the "
                "account the org key should act on) before making Stripe calls."
            )
        stripe_api_version = getattr(stripe, "STRIPE_API_VERSION", None) or "2024-06-20"
        headers = {
            "Stripe-Version": stripe_api_version,
            "Stripe-Context": self.stripe_account,
        }
        if hasattr(stripe, "set_default_headers"):
            stripe.set_default_headers(headers)
            logger.info(
                "Configured Stripe Organization key (sk_org_) with "
                "Stripe-Context=%s, Stripe-Version=%s",
                self.stripe_account,
                stripe_api_version,
            )
        else:
            logger.warning(
                "Installed Stripe SDK lacks set_default_headers; cannot set "
                "organization-key headers (Stripe-Version/Stripe-Context). "
                "Upgrade to stripe>=5.0.0 (pip install -U stripe), or use an "
                "account-level rk_live_ key instead."
            )

    def spend_profits(self, amount_cents, currency="usd",
                      description="Spending profits for Oscar Broome",
                      destination_account=None):
        """
        Create a payment or transfer to spend profits through Stripe.

        Parameters:
        - amount_cents: int, amount in cents to spend
        - currency: str, currency code (default "usd")
        - description: str, description for the payment
        - destination_account: str or None, Stripe connected account ID to
          transfer funds to (optional)

        Returns:
        - dict: Stripe payment or transfer object (or a simulated result when
          the Stripe SDK / credentials are unavailable).
        """
        if not self.stripe_available:
            # Stripe SDK/credentials unavailable: simulate the operation so the
            # quantum financial flows keep running end-to-end without crashing.
            logger.info(
                "Stripe unavailable; simulating spend_profits(%s %s)",
                amount_cents, currency)
            return {
                "status": "succeeded_simulated",
                "amount": amount_cents,
                "currency": currency,
                "description": description,
                "destination_account": destination_account,
            }
        if destination_account:
            # Create a transfer to a connected account
            transfer = stripe.Transfer.create(
                amount=amount_cents,
                currency=currency,
                destination=destination_account,
                description=description,
            )
            return transfer
        else:
            # Create a payment intent to charge the account (simulate spending)
            payment_intent = stripe.PaymentIntent.create(
                amount=amount_cents,
                currency=currency,
                payment_method_types=["card"],
                description=description,
            )
            return payment_intent
