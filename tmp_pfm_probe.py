from test_runner import app

def main():
    c = app.test_client()

    get_resp = c.get("/pfm/bills", query_string={"user_id": "probe_user"})
    print("GET /pfm/bills:", get_resp.status_code, get_resp.get_json())

    post_payload = {
        "user_id": "probe_user",
        "name": "Internet Bill",
        "amount": 80.0,
        "due_date": "2026-12-31",
        "category": "utilities",
        "frequency": "monthly",
    }
    post_resp = c.post("/pfm/bills", json=post_payload)
    print("POST /pfm/bills:", post_resp.status_code, post_resp.get_json())

    cat_payload = {
        "user_id": "probe_user",
        "transactions": [
            {
                "transaction_id": "t1",
                "description": "Netflix Subscription",
                "amount": -15.99,
            }
        ],
    }
    cat_resp = c.post("/pfm/transactions/categorize", json=cat_payload)
    print("POST /pfm/transactions/categorize:", cat_resp.status_code, cat_resp.get_json())

if __name__ == "__main__":
    main()
