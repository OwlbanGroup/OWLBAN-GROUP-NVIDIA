import os
import stripe

class StripeIntegration:
    def __init__(self):
        # Load Stripe API key from environment variable
        self.api_key = os.getenv("STRIPE_API_KEY")
        if not self.api_key:
            # Use test key for development/demo purposes
            self.api_key = "sk_test_dummy_key_for_development"
            print("WARNING: Using dummy Stripe API key for development. Set STRIPE_API_KEY environment variable for production.")
        stripe.api_key = self.api_key

    def spend_profits(self, amount_cents, currency="usd", description="Spending profits for Oscar Broome", destination_account=None):
        """
        Create a payment or transfer to spend profits through Stripe.

        Parameters:
        - amount_cents: int, amount in cents to spend
        - currency: str, currency code (default "usd")
        - description: str, description for the payment
        - destination_account: str or None, Stripe connected account ID to transfer funds to (optional)

        Returns:
        - dict: Stripe payment or transfer object
        """
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
