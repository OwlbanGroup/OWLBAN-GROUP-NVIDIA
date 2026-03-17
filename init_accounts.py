#!/usr/bin/env python3
\"\"\"Seed script for sample bank accounts.\"\"\"
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database_fixed import db_manager
from src.banking_service import banking_service
from src.banking_data_models import AccountType

SAMPLE_ACCOUNTS = [
    {
        'user_id': 'user1',
        'account_type': 'checking',
        'initial_balance': 5000.0,
    },
    {
        'user_id': 'user1',
        'account_type': 'savings',
        'initial_balance': 15000.0,
    },
    {
        'user_id': 'user2',
        'account_type': 'checking',
        'initial_balance': 2500.0,
    },
]

if __name__ == '__main__':
    print(\"Initializing database...\")
    db_manager.initialize_database()
    
    print(\"Seeding sample accounts...\")
    created = []
    for data in SAMPLE_ACCOUNTS:
        try:
            account = banking_service.create_account(data)
            created.append(account.to_dict())
            print(f\"Created: {account.account_number} (${data['initial_balance']}) for {data['user_id']}\")
        except Exception as e:
            print(f\"Failed to create account for {data['user_id']}: {e}\")
    
    print(f\"\n✅ Seeded {len(created)} sample accounts.\")
    print(\"Test with: curl -H 'Authorization: Bearer user1' http://localhost:5000/banking/accounts\")

